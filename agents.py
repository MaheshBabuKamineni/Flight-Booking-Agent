"""AutoGen multi-agent team for flight booking, shared by the console app (main.py)
and the Streamlit UI (app.py).

Agents (coordinated by a SelectorGroupChat, which uses the LLM to pick who speaks next):
  - travel_coordinator : understands the request, delegates, and presents the final plan
  - timing_advisor     : recommends the best day / time of day to fly
  - flight_agent       : searches flights and books the one the user confirms
  - stay_agent         : recommends hotels and motels near the airport or city centre
  - user               : (console mode only) the human, via UserProxyAgent
"""

import json
import os
from collections.abc import Sequence
from datetime import date
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # picks up GROQ_API_KEY / GROQ_MODEL from a .env file if present

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core import FunctionCall
from autogen_core.models import ModelInfo
from autogen_core.models import FunctionExecutionResult
from autogen_ext.models.openai import OpenAIChatCompletionClient

from travel_tools import book_flight, recommend_best_travel_time, search_flights, search_nearby_stays

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"
AGENT_NAMES = ["travel_coordinator", "timing_advisor", "flight_agent", "stay_agent"]
NEXT_AGENT = {
    "travel_coordinator": "timing_advisor",
    "timing_advisor": "flight_agent",
    "flight_agent": "stay_agent",
    "stay_agent": "travel_coordinator",
}
DONE_MARKER = "TERMINATE"
ASK_USER_MARKER = "NEED_USER_INPUT"  # UI mode: an agent ends its turn with this to hand back to the human


def select_next_speaker(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str:
    """Route agents in a fixed order without spending an LLM call on selection."""
    recent_sources = [
        getattr(message, "source", None)
        for message in reversed(messages)
        if getattr(message, "source", None) in (*AGENT_NAMES, "user")
    ]
    if not recent_sources:
        return "travel_coordinator"

    last_source = recent_sources[0]
    if last_source == "user":
        last_source = next((source for source in recent_sources[1:] if source in AGENT_NAMES), None)
        if last_source is None:
            return "travel_coordinator"
        return last_source

    return NEXT_AGENT[last_source]


def format_model_error(exc: Exception) -> str:
    """Extract provider details from an API error, including wrapped exceptions."""
    current: BaseException | None = exc
    api_error: BaseException | None = None
    while current is not None:
        if getattr(current, "status_code", None) is not None:
            api_error = current
            break
        current = current.__cause__ or current.__context__

    if api_error is None:
        error_text = str(exc)
        if "RateLimitError" in error_text or "Error code: 429" in error_text:
            return (
                "Groq rate limit (429): Groq is limiting requests for this model or account. "
                "Wait for the quota window to reset, then try again or check your Groq account limits."
            )
        return f"{type(exc).__name__}: {error_text}"

    status_code = getattr(api_error, "status_code", None)
    body: Any = getattr(api_error, "body", None)
    response = getattr(api_error, "response", None)
    if not isinstance(body, dict):
        response_text = getattr(response, "text", None)
        if response_text:
            try:
                body = json.loads(response_text)
            except json.JSONDecodeError:
                body = response_text

    if isinstance(body, dict):
        detail: Any = body.get("error", body)
        if isinstance(detail, dict):
            message = detail.get("message") or detail.get("code") or str(detail)
            status = detail.get("status")
            detail_text = f"{message} ({status})" if status else str(message)
        else:
            detail_text = str(detail)
    elif body:
        detail_text = str(body)
    else:
        detail_text = str(api_error)

    metadata = []
    headers = getattr(response, "headers", {})
    retry_after = headers.get("retry-after") if headers else None
    request_id = getattr(api_error, "request_id", None)
    if retry_after:
        metadata.append(f"Retry-After: {retry_after}")
    if request_id:
        metadata.append(f"Request ID: {request_id}")
    metadata_suffix = f" ({'; '.join(metadata)})" if metadata else ""
    return f"Groq API error ({status_code}): {detail_text}{metadata_suffix}"


def make_model_client(model: str = DEFAULT_MODEL) -> OpenAIChatCompletionClient:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set GROQ_API_KEY in your environment or .env file before starting the assistant.")
    if not api_key.isascii():
        raise ValueError(
            "GROQ_API_KEY contains non-ASCII characters. Paste the API key from Groq Console "
            "without emojis or other copied text."
        )

    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=GROQ_API_BASE_URL,
        timeout=60,
        max_retries=1,
        include_name_in_message=False,
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family="unknown",
        ),
    )


def summarize_tool_result(call: FunctionCall, result: FunctionExecutionResult) -> str:
    """Keep tool output compact for the UI and subsequent model prompts."""
    if result.is_error:
        return f"{call.name} failed: {result.content}"

    try:
        data = json.loads(result.content)
    except (TypeError, json.JSONDecodeError):
        return f"{call.name}: {result.content}"

    if not isinstance(data, dict):
        return f"{call.name}: {json.dumps(data, ensure_ascii=True)}"
    if "error" in data:
        return f"{call.name} error: {data['error']}"

    if call.name == "search_flights":
        flights = [
            {key: flight[key] for key in ("flight_id", "airline", "departure", "duration", "stops", "price_usd")}
            for flight in data.get("flights", [])[:3]
        ]
        summary = {"count": data.get("count", 0), "top_flights": flights}
    elif call.name == "search_nearby_stays":
        stays = [
            {key: stay[key] for key in (
                "name", "type", "price_per_night_usd", "total_price_usd", "guest_rating", "value_score"
            )}
            for stay in data.get("stays", [])[:5]
        ]
        summary = {"city": data.get("city"), "nights": data.get("nights"), "count": data.get("count", 0),
                   "top_stays": stays}
    elif call.name == "recommend_best_travel_time":
        summary = {
            "recommended_flight": data.get("recommended_flight"),
            "cheapest_days": data.get("cheapest_days"),
            "most_expensive_day": data.get("most_expensive_day"),
            "cheapest_time_slots": (data.get("time_slots_by_avg_fare") or [])[:3],
            "tips": data.get("tips"),
        }
    elif call.name == "book_flight":
        summary = {
            "status": data.get("status"),
            "message": "Booking details are private and are not shared with the agent.",
        }
    else:
        summary = data

    return f"{call.name}: {json.dumps(summary, ensure_ascii=True)}"


def build_team(model_client: OpenAIChatCompletionClient, ui_mode: bool = False) -> SelectorGroupChat:
    """Build the team. In ui_mode there is no UserProxyAgent: when an agent needs the
    human it ends its message with NEED_USER_INPUT, the run stops, and the next chat
    message from the UI continues the same team (AutoGen keeps the conversation state)."""
    today = date.today().isoformat()
    if ui_mode:
        ask_rule = (f"When you need information or a confirmation from the traveller, ask one concise "
                    f"question and end your message with the word {ASK_USER_MARKER}.")
    else:
        ask_rule = "When you need information or a confirmation from the traveller, ask one concise question."

    coordinator = AssistantAgent(
        name="travel_coordinator",
        model_client=model_client,
        description="Lead travel planner. Gathers requirements, delegates to specialists, presents the final itinerary.",
        system_message=f"""You are the lead travel coordinator. Today is {today}.
Work out origin, destination, a date or date window, number of nights, budget and preferences.
If something essential is missing, ask the traveller. {ask_rule}
Delegate: timing_advisor for the best time to fly, flight_agent for flights/booking,
stay_agent for hotels and motels. Do not invent flights, prices or hotels yourself.
When the work is done, present a clear summary: recommended flight and booking status (if booked),
best time to fly and why, and the top 3 stays with price, rating and distance.
Never identify a guest or disclose a passenger name, email, or reservation code based on a
booking reference. If asked who a code belongs to, say you cannot disclose private booking details.
End that final summary with the word {DONE_MARKER}.""",
    )

    timing_advisor = AssistantAgent(
        name="timing_advisor",
        model_client=model_client,
        tools=[recommend_best_travel_time],
        reflect_on_tool_use=False,
        tool_call_summary_formatter=summarize_tool_result,
        description="Finds the best day and time of day to fly using fare trends across a date window.",
        system_message=f"""You recommend WHEN to fly. Today is {today}.
Use recommend_best_travel_time over the traveller's acceptable date window (if they gave a single
date, scan 3 days either side). Report the recommended day, the cheapest time slot, how much
can be saved versus the most expensive day, and the booking tips. Be brief.""",
    )

    flight_agent = AssistantAgent(
        name="flight_agent",
        model_client=model_client,
        tools=[search_flights, book_flight],
        reflect_on_tool_use=False,
        tool_call_summary_formatter=summarize_tool_result,
        description="Searches flights for a specific date and books a flight after the traveller confirms.",
        system_message=f"""You handle flights. Today is {today}.
Use search_flights and show the top 3 options (flight_id, airline, departure, duration, stops, price).
NEVER call book_flight until the traveller has explicitly said yes to a specific flight_id and given
the passenger's full name and email. {ask_rule}""",
    )

    stay_agent = AssistantAgent(
        name="stay_agent",
        model_client=model_client,
        tools=[search_nearby_stays],
        reflect_on_tool_use=False,
        tool_call_summary_formatter=summarize_tool_result,
        description="Recommends nearby hotels and motels at the destination for the travel dates.",
        system_message=f"""You recommend places to stay. Today is {today}.
Check-in is the flight arrival date. Use search_nearby_stays near the airport for short or
late-arrival stays, near the city centre otherwise (or whatever the traveller asked for).
Include both hotels and motels unless the traveller has a preference. Show the top 3-5 with
type, price per night, total, rating, distance and key amenities, and say which is best value.""",
    )

    participants = [coordinator, timing_advisor, flight_agent, stay_agent]
    termination = TextMentionTermination(DONE_MARKER, sources=AGENT_NAMES) | MaxMessageTermination(40)

    if ui_mode:
        termination = termination | TextMentionTermination(ASK_USER_MARKER, sources=AGENT_NAMES)
        user_rule = ""
    else:
        participants.append(UserProxyAgent(
            name="user",
            description="The human traveller. Select when a question needs their answer or a booking needs their confirmation.",
            input_func=input,
        ))
        termination = termination | TextMentionTermination("exit", sources=["user"])
        user_rule = "- If the last message asks the traveller a question or needs booking confirmation, choose user.\n"

    selector_prompt = f"""You are choosing who speaks next in a travel-booking team.

{{roles}}

Conversation so far:
{{history}}

Rules:
{user_rule}- Choose a specialist when their part of the work is still missing.
- Choose travel_coordinator to kick things off and to write the final summary.
Read the conversation, then select exactly one role from {{participants}}. Reply with the name only."""

    return SelectorGroupChat(
        participants=participants,
        model_client=model_client,
        selector_prompt=selector_prompt,
        selector_func=select_next_speaker,
        termination_condition=termination,
        allow_repeated_speaker=True,
    )
