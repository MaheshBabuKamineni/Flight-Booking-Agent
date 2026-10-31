"""Privacy policy checks that do not require a model API key."""

import asyncio
import unittest
import json

from autogen_core import FunctionCall
from autogen_core.models import FunctionExecutionResult
from agents import summarize_tool_result
from privacy_guardrails import (
    PRIVATE_OUTPUT,
    PrivacyGuardrails,
    contains_private_booking_code,
    is_reservation_owner_lookup,
)


class PrivacyGuardrailTests(unittest.TestCase):
    def test_blocks_owner_lookup_with_typo_and_booking_code(self) -> None:
        self.assertTrue(is_reservation_owner_lookup("Who is having Rseration code ABC-1234?"))

    def test_blocks_owner_lookup_with_pnr(self) -> None:
        self.assertTrue(is_reservation_owner_lookup("Who does PNR: ABC123 belong to?"))

    def test_allows_general_trip_planning(self) -> None:
        self.assertFalse(is_reservation_owner_lookup("Plan a trip from Hyderabad to Dubai."))

    def test_detects_hyphenated_booking_code(self) -> None:
        self.assertTrue(contains_private_booking_code("Reservation code ABC-1234"))

    def test_detects_labeled_six_character_pnr(self) -> None:
        self.assertTrue(contains_private_booking_code("PNR: ABC123"))

    def test_does_not_match_flight_identifier(self) -> None:
        self.assertFalse(contains_private_booking_code("Flight HYDDUB-1110-1"))

    def test_nemo_output_rail_redacts_person_and_contact_details(self) -> None:
        async def sanitize() -> str:
            guardrails = PrivacyGuardrails()
            try:
                return await guardrails.sanitize_output(
                    "Guest Jane Example can be reached at jane@example.com or 202-555-0148."
                )
            finally:
                await guardrails.close()

        result = asyncio.run(sanitize())

        self.assertNotIn("Jane Example", result)
        self.assertNotIn("jane@example.com", result)
        self.assertNotIn("202-555-0148", result)
        self.assertIn("[PERSON]", result)
        self.assertIn("[EMAIL]", result)
        self.assertIn("[PHONE]", result)

    def test_nemo_output_rail_withholds_booking_code(self) -> None:
        async def sanitize() -> str:
            guardrails = PrivacyGuardrails()
            try:
                return await guardrails.sanitize_output("Reservation code ABC-1234")
            finally:
                await guardrails.close()

        result = asyncio.run(sanitize())

        self.assertEqual(result, PRIVATE_OUTPUT)
        self.assertNotIn("ABC-1234", result)

    def test_nemo_output_rail_redacts_financial_identifiers(self) -> None:
        async def sanitize() -> str:
            guardrails = PrivacyGuardrails()
            try:
                return await guardrails.sanitize_output(
                    "Card 4111 1111 1111 1111, SSN 123-45-6789, IBAN GB82 WEST 1234 5698 7654 32."
                )
            finally:
                await guardrails.close()

        result = asyncio.run(sanitize())

        self.assertNotIn("4111 1111 1111 1111", result)
        self.assertNotIn("123-45-6789", result)
        self.assertNotIn("GB82 WEST 1234 5698 7654 32", result)
        self.assertIn("[CREDIT_CARD]", result)
        self.assertIn("[SSN]", result)
        self.assertIn("[IBAN]", result)

    def test_booking_summary_does_not_include_private_details(self) -> None:
        call = FunctionCall(id="call-1", name="book_flight", arguments="{}")
        result = FunctionExecutionResult(
            content=json.dumps({
                "status": "CONFIRMED",
                "pnr": "ABC123",
                "passenger": "Example Passenger",
                "email": "passenger@example.com",
            }),
            call_id="call-1",
            is_error=False,
            name="book_flight",
        )

        summary = summarize_tool_result(call, result)

        self.assertIn("CONFIRMED", summary)
        self.assertNotIn("ABC123", summary)
        self.assertNotIn("Example Passenger", summary)
        self.assertNotIn("passenger@example.com", summary)


if __name__ == "__main__":
    unittest.main()
