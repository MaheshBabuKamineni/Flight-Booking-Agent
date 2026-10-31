r"""Console version of the flight booking assistant.

Run:
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    $env:GROQ_API_KEY="your-groq-api-key"     (PowerShell)
    .venv\Scripts\python.exe main.py

For the web UI, run:  .venv\Scripts\python.exe -m streamlit run app.py
"""

import asyncio

from autogen_agentchat.ui import Console

from agents import build_team, format_model_error, make_model_client
from privacy_guardrails import PRIVACY_REFUSAL, PrivacyGuardrails, is_reservation_owner_lookup


async def sanitized_stream(stream, privacy_guardrails: PrivacyGuardrails):
    async for item in stream:
        content = getattr(item, "content", None)
        if isinstance(content, str):
            item = item.model_copy(update={"content": await privacy_guardrails.sanitize_output(content)})
        elif type(item).__name__ == "ToolCallExecutionEvent" and isinstance(content, list):
            results = [
                result.model_copy(
                    update={"content": await privacy_guardrails.sanitize_output(result.content)}
                )
                for result in content
            ]
            item = item.model_copy(update={"content": results})
        elif type(item).__name__ == "ToolCallRequestEvent" and isinstance(content, list):
            calls = [
                call.model_copy(update={"arguments": "[PRIVATE BOOKING DETAILS REDACTED]"})
                if call.name == "book_flight"
                else call
                for call in content
            ]
            item = item.model_copy(update={"content": calls})
        yield item


async def main() -> None:
    privacy_guardrails = PrivacyGuardrails()
    model_client = None
    try:
        model_client = make_model_client()
        print("Flight Booking Assistant (type 'exit' when asked for input to quit)\n")
        task = input("Where and when do you want to travel?\n> ").strip() or (
            "I want to fly from Hyderabad to Dubai sometime in the next 3 weeks for 4 nights. "
            "Find the best time to fly and a good hotel or motel near the airport under $150/night."
        )
        if is_reservation_owner_lookup(task):
            print(PRIVACY_REFUSAL)
            return
        await Console(sanitized_stream(build_team(model_client).run_stream(task=task), privacy_guardrails))
    except Exception as exc:
        print(format_model_error(exc))
    finally:
        if model_client is not None:
            await model_client.close()
        await privacy_guardrails.close()


if __name__ == "__main__":
    asyncio.run(main())
