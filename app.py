r"""Streamlit front end for the AutoGen flight booking assistant.

Run:
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    $env:GROQ_API_KEY="your-groq-api-key"     (PowerShell)
    .venv\Scripts\python.exe -m streamlit run app.py
"""

import asyncio
import logging
import os
import queue
import threading
from datetime import date, timedelta

import streamlit as st
from autogen_agentchat.base import TaskResult

from agents import (
    AGENT_NAMES,
    ASK_USER_MARKER,
    DEFAULT_MODEL,
    DONE_MARKER,
    build_team,
    format_model_error,
    make_model_client,
)
from privacy_guardrails import PRIVACY_REFUSAL, PrivacyGuardrails, is_reservation_owner_lookup

logger = logging.getLogger(__name__)

AVATARS = {"travel_coordinator": "🧭", "timing_advisor": "⏰", "flight_agent": "✈️", "stay_agent": "🏨", "user": "🧑"}
LABELS = {"travel_coordinator": "Travel Coordinator", "timing_advisor": "Timing Advisor",
          "flight_agent": "Flight Agent", "stay_agent": "Stay Agent", "user": "You"}

st.set_page_config(page_title="AI Flight Booking Assistant", page_icon="✈️", layout="wide")


# ---------------------------------------------------------------------------
# Async plumbing: AutoGen is async and its model client is bound to one event
# loop, while Streamlit re-runs this script on every interaction. So each browser
# session gets one long-lived event loop in a background thread, and the team
# (with its conversation memory) lives on that loop.
# ---------------------------------------------------------------------------
class AgentSession:
    def __init__(self, model: str):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.privacy_guardrails = PrivacyGuardrails()
        self.model_client = self._run(self._make_client(model))
        self.team = build_team(self.model_client, ui_mode=True)

    @staticmethod
    async def _make_client(model: str):
        return make_model_client(model)

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def sanitize_output(self, text: str) -> str:
        return self._run(self.privacy_guardrails.sanitize_output(text))

    def stream(self, task: str):
        """Yield agent messages/events as the team produces them."""
        q: queue.Queue = queue.Queue()

        async def pump():
            try:
                async for item in self.team.run_stream(task=task):
                    q.put(item)
            except Exception as exc:  # surface API/tool errors in the UI
                q.put(exc)
            finally:
                q.put(None)

        asyncio.run_coroutine_threadsafe(pump(), self.loop)
        while (item := q.get()) is not None:
            yield item

    def close(self):
        try:
            self._run(self._close_resources())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)

    async def _close_resources(self):
        try:
            await self.model_client.close()
        finally:
            await self.privacy_guardrails.close()


def clean(text: str) -> str:
    return text.replace(ASK_USER_MARKER, "").replace(DONE_MARKER, "").strip()


def render(entry: dict) -> None:
    with st.chat_message(entry["source"], avatar=AVATARS.get(entry["source"], "🤖")):
        st.markdown(f"**{LABELS.get(entry['source'], entry['source'])}**")
        if entry["kind"] == "text":
            st.markdown(entry["content"])
        else:
            with st.expander(entry["title"]):
                st.code(entry["content"], language="json")


def to_entry(item) -> dict | None:
    """Convert an AutoGen message/event into something displayable (or None to skip)."""
    source = getattr(item, "source", "")
    kind = type(item).__name__
    if source not in AGENT_NAMES:
        return None  # the user's own message is shown when they send it
    if kind == "ToolCallRequestEvent":
        calls = "\n".join(f"{c.name}({c.arguments})" for c in item.content)
        return {"source": source, "kind": "tool", "title": "🔧 Calling tools", "content": calls}
    if kind == "ToolCallExecutionEvent":
        results = "\n\n".join(r.content for r in item.content)
        return {"source": source, "kind": "tool", "title": "📦 Tool results", "content": results}
    if kind in ("TextMessage", "ToolCallSummaryMessage") and isinstance(item.content, str):
        text = clean(item.content)
        return {"source": source, "kind": "text", "content": text} if text else None
    return None


def start_session(model: str) -> None:
    if "agent_session" in st.session_state:
        st.session_state.agent_session.close()
    st.session_state.agent_session = AgentSession(model)
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Sidebar: settings + quick trip planner form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.text_input("Groq model", value=DEFAULT_MODEL)
    configured_key = os.getenv("GROQ_API_KEY", "").strip()
    if not configured_key or not configured_key.isascii():
        key = st.text_input(
            "Groq API key",
            type="password",
            help="Get a key from Groq Console, or set GROQ_API_KEY before launching.",
        )
        if key:
            os.environ["GROQ_API_KEY"] = key.strip()
    if st.button("🔄 New conversation", use_container_width=True):
        configured_key = os.getenv("GROQ_API_KEY", "").strip()
        if configured_key and configured_key.isascii():
            start_session(model)
            st.rerun()
        else:
            st.warning("Enter a valid ASCII Groq API key above before starting a conversation.")

    st.divider()
    st.header("🧳 Quick trip planner")
    with st.form("trip"):
        origin = st.text_input("From", "Hyderabad")
        destination = st.text_input("To", "Dubai")
        window = st.date_input("Flexible departure window",
                               (date.today() + timedelta(days=14), date.today() + timedelta(days=28)),
                               min_value=date.today())
        nights = st.number_input("Nights", 1, 60, 4)
        preference = st.selectbox("Flight priority", ["balanced", "cheapest", "fastest"])
        stay_type = st.selectbox("Stay type", ["any", "hotel", "motel"])
        near = st.selectbox("Stay near", ["airport", "city_center"])
        budget = st.slider("Max price per night (USD)", 30, 500, 150, step=10)
        planned = st.form_submit_button("Plan my trip ✈️", use_container_width=True)

prompt_from_form = None
if planned:
    start, end = (window if isinstance(window, tuple) and len(window) == 2 else (window, window))
    prompt_from_form = (
        f"Plan a trip from {origin} to {destination}. I can depart any day between {start} and {end} "
        f"and will stay {nights} nights. Flight priority: {preference}. Recommend the best time to fly, "
        f"show the best flights, and suggest {stay_type if stay_type != 'any' else 'hotels and motels'} "
        f"near the {near.replace('_', ' ')} under ${budget}/night."
    )

# ---------------------------------------------------------------------------
# Main chat
# ---------------------------------------------------------------------------
st.title("✈️ AI Flight Booking Assistant")
st.caption("A multi-agent AutoGen team powered by Groq: it finds the best time to fly, searches flights, "
           "books after your confirmation, and recommends nearby hotels and motels.")

configured_key = os.getenv("GROQ_API_KEY", "").strip()
api_key_configured = bool(configured_key) and configured_key.isascii()
if api_key_configured and "agent_session" not in st.session_state:
    start_session(model)

if not api_key_configured:
    if configured_key:
        st.error(
            "GROQ_API_KEY contains non-ASCII characters. Replace it with the API key from Groq Console "
            "and remove any emoji or extra copied text."
        )
    else:
        st.info(
            "To start chatting, add a Groq API key in the sidebar or configure GROQ_API_KEY in your .env file. "
            "Get a key from Groq Console."
        )

for entry in st.session_state.get("history", []):
    try:
        entry["content"] = st.session_state.agent_session.sanitize_output(entry["content"])
    except Exception:
        logger.exception("NeMo privacy guardrails failed; chat history was withheld.")
        st.error("Privacy check failed; chat history was withheld.")
        st.stop()
    render(entry)

user_text = st.chat_input(
    "e.g. Book me a flight from Delhi to Singapore next month for 5 nights",
    disabled=not api_key_configured,
)
task = prompt_from_form or user_text

if task:
    if not api_key_configured:
        st.error("Enter a valid Groq API key in the sidebar (or set GROQ_API_KEY) first.")
        st.stop()
    if is_reservation_owner_lookup(task):
        refusal = {"source": "travel_coordinator", "kind": "text", "content": PRIVACY_REFUSAL}
        st.session_state.history.append(refusal)
        render(refusal)
        st.stop()
    user_entry = {"source": "user", "kind": "text", "content": task}
    try:
        user_entry["content"] = st.session_state.agent_session.sanitize_output(user_entry["content"])
    except Exception:
        logger.exception("NeMo privacy guardrails failed; user message was withheld.")
        st.error("Privacy check failed; the message was not sent.")
        st.stop()
    st.session_state.history.append(user_entry)
    render(user_entry)

    with st.spinner("Agents are working..."):
        for item in st.session_state.agent_session.stream(task):
            if isinstance(item, Exception):
                st.error(format_model_error(item))
                break
            if isinstance(item, TaskResult):
                if item.stop_reason and "Maximum number of messages" in item.stop_reason:
                    st.warning("The agents hit the message limit. Send a message to continue.")
                continue
            entry = to_entry(item)
            if entry:
                try:
                    entry["content"] = st.session_state.agent_session.sanitize_output(entry["content"])
                except Exception:
                    logger.exception("NeMo privacy guardrails failed; assistant output was withheld.")
                    st.error("Privacy check failed; this response was withheld.")
                    break
                st.session_state.history.append(entry)
                render(entry)
