"""Privacy checks for reservation-owner requests and assistant output."""

import re
from pathlib import Path

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

PRIVACY_REFUSAL = (
    "I can’t identify a guest or disclose reservation details from a booking code. "
    "Please check your private booking confirmation or contact the travel provider."
)
PRIVATE_OUTPUT = "Private booking information was withheld."

_BOOKING_CONTEXT = re.compile(
    r"\b(?:reserv\w*|rser\w*|booking|pnr|ticket|confirmation)\b", re.IGNORECASE
)
_OWNER_REQUEST = re.compile(
    r"\b(?:who|whose|owner|belongs?\s+to|guest|passenger|name\s+of|identify)\b",
    re.IGNORECASE,
)
_BOOKING_CODE = re.compile(
    r"\b(?:[A-Z]{2,6}-\d{3,8}(?!-[A-Z0-9])|"
    r"(?:PNR|(?:booking|reservation|confirmation)\s+"
    r"(?:code|number|reference|id))\s*[:#-]?\s*[A-Z0-9]{5,10})\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
_US_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[\s-]?[A-Z0-9]){11,30}\b", re.IGNORECASE)
_CARD_CANDIDATE = re.compile(r"(?<!\w)(?:\d[ -]?){13,19}(?!\w)")
_PERSON_CONTEXTS = (
    re.compile(
        r"(?i:\b(?:guest|passenger|travell?er|reservation holder|booking holder|"
        r"account holder)\s+(?:name\s*(?:is|:)\s*)?)"
        r"(?P<name>[A-Z][a-z]+(?:[\s\u00a0\u202f-]+[A-Z][a-z]+){0,3})"
    ),
    re.compile(
        r"(?i:\b(?:belongs to|owned by|booked for|name is)\s+)"
        r"(?P<name>[A-Z][a-z]+(?:[\s\u00a0\u202f-]+[A-Z][a-z]+){0,3})"
    ),
)


def is_reservation_owner_lookup(text: str) -> bool:
    """Return whether a message asks to identify a guest from a reservation reference."""
    has_booking_context = bool(_BOOKING_CONTEXT.search(text)) or bool(_BOOKING_CODE.search(text))
    return has_booking_context and bool(_OWNER_REQUEST.search(text))


def contains_private_booking_code(text: str) -> bool:
    return bool(_BOOKING_CODE.search(text))


def _is_luhn_valid(candidate: str) -> bool:
    digits = [int(char) for char in candidate if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for offset, digit in enumerate(reversed(digits)):
        if offset % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def redact_sensitive_data(source: str, text: str) -> str:
    """Redact common PII locally so output checks need no external NLP model."""
    del source
    redacted = _EMAIL.sub("[EMAIL]", text)
    redacted = _US_SSN.sub("[SSN]", redacted)
    redacted = _IBAN.sub("[IBAN]", redacted)
    redacted = _CARD_CANDIDATE.sub(
        lambda match: "[CREDIT_CARD]" if _is_luhn_valid(match.group()) else match.group(),
        redacted,
    )
    redacted = _PHONE.sub("[PHONE]", redacted)
    for pattern in _PERSON_CONTEXTS:
        redacted = pattern.sub("[PERSON]", redacted)
    return redacted


class PrivacyGuardrails:
    """Use a NeMo output rail to redact common PII in assistant-visible text."""

    def __init__(self) -> None:
        config_path = Path(__file__).parent / "guardrails" / "privacy"
        config = RailsConfig.from_path(str(config_path))
        self._rails = Guardrails(config, use_iorails=False)
        self._rails.register_action(redact_sensitive_data, name="mask_sensitive_data")

    async def sanitize_output(self, text: str) -> str:
        result = await self._rails.check_async(
            messages=[{"role": "assistant", "content": text}],
            rail_types=[RailType.OUTPUT],
        )
        if result.status == RailStatus.BLOCKED:
            return PRIVATE_OUTPUT

        if contains_private_booking_code(result.content):
            return PRIVATE_OUTPUT
        return result.content

    async def close(self) -> None:
        await self._rails.shutdown()
