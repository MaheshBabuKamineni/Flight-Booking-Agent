"""Travel tools used by the agents.

These return realistic mock data so the system runs without any paid API keys.
To go live, replace the body of each function with calls to a real provider
(e.g. Amadeus / Skyscanner / Duffel for flights, Booking.com / Google Places /
Amadeus Hotel Search for stays). Keep the same signatures and return shapes and
the agents keep working unchanged.
"""

from __future__ import annotations

import hashlib
import random
import uuid
from datetime import date, datetime, timedelta
from typing import Annotated, Literal

AIRLINES = ["IndiGo", "Air India", "Vistara", "Emirates", "Lufthansa", "United", "Delta", "Qatar Airways"]

# Departure windows with a typical demand multiplier (busy slots cost more).
TIME_SLOTS = {
    "early_morning (05:00-08:00)": (5, 8, 0.85),
    "morning (08:00-12:00)": (8, 12, 1.15),
    "afternoon (12:00-17:00)": (12, 17, 1.00),
    "evening (17:00-21:00)": (17, 21, 1.20),
    "red_eye (21:00-05:00)": (21, 29, 0.78),
}

# Day-of-week demand (Mon=0). Tue/Wed/Sat are usually cheapest.
WEEKDAY_FACTOR = [1.05, 0.88, 0.90, 1.00, 1.18, 0.95, 1.12]

BOOKINGS: dict[str, dict] = {}
FLIGHT_INDEX: dict[str, dict] = {}  # every flight shown to an agent, so it can be booked by id


def _rng(*parts: object) -> random.Random:
    """Deterministic RNG so the same query always returns the same results."""
    seed = hashlib.sha256("|".join(str(p).lower() for p in parts).encode()).hexdigest()
    return random.Random(int(seed[:16], 16))


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _base_fare(origin: str, destination: str) -> int:
    return _rng("base", *sorted([origin, destination])).randint(90, 650)


def _flights_for_day(origin: str, destination: str, day: date) -> list[dict]:
    rng = _rng("flights", origin, destination, day.isoformat())
    base = _base_fare(origin, destination)
    days_out = max((day - date.today()).days, 0)
    # Booking 21-60 days ahead is usually the sweet spot; last-minute is expensive.
    advance_factor = 1.45 if days_out < 7 else 1.15 if days_out < 21 else 0.92 if days_out <= 60 else 1.0

    flights = []
    for i in range(rng.randint(6, 10)):
        slot_name, (start_h, end_h, slot_factor) = rng.choice(list(TIME_SLOTS.items()))
        hour = rng.randint(start_h, end_h - 1) % 24
        minute = rng.choice([0, 15, 30, 45])
        stops = rng.choices([0, 1, 2], weights=[5, 4, 1])[0]
        duration_min = rng.randint(80, 600) + stops * rng.randint(60, 180)
        dep = datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=minute)
        price = base * WEEKDAY_FACTOR[day.weekday()] * slot_factor * advance_factor
        price *= (0.85 if stops else 1.0) * rng.uniform(0.9, 1.12)
        flights.append({
            "flight_id": f"{origin[:3].upper()}{destination[:3].upper()}-{day:%m%d}-{i + 1}",
            "airline": rng.choice(AIRLINES),
            "origin": origin,
            "destination": destination,
            "departure": dep.strftime("%Y-%m-%d %H:%M"),
            "arrival": (dep + timedelta(minutes=duration_min)).strftime("%Y-%m-%d %H:%M"),
            "time_slot": slot_name,
            "duration": f"{duration_min // 60}h {duration_min % 60}m",
            "stops": stops,
            "price_usd": round(price, 2),
            "seats_left": rng.randint(1, 30),
        })
    flights = [FLIGHT_INDEX.setdefault(f["flight_id"], f) for f in flights]
    return sorted(flights, key=lambda f: f["price_usd"])


def search_flights(
    origin: Annotated[str, "Origin city or IATA code, e.g. 'Hyderabad' or 'HYD'"],
    destination: Annotated[str, "Destination city or IATA code"],
    travel_date: Annotated[str, "Departure date in YYYY-MM-DD format"],
    max_stops: Annotated[int, "Maximum number of stops allowed (0 = non-stop only)"] = 2,
) -> dict:
    """Search available flights for one day, sorted by price (cheapest first)."""
    try:
        day = _parse_date(travel_date)
    except ValueError:
        return {"error": f"Invalid date '{travel_date}'. Use YYYY-MM-DD."}
    if day < date.today():
        return {"error": "Travel date is in the past."}
    flights = [f for f in _flights_for_day(origin, destination, day) if f["stops"] <= max_stops]
    return {"count": len(flights), "flights": flights}


def recommend_best_travel_time(
    origin: Annotated[str, "Origin city or IATA code"],
    destination: Annotated[str, "Destination city or IATA code"],
    earliest_date: Annotated[str, "Earliest acceptable departure date, YYYY-MM-DD"],
    latest_date: Annotated[str, "Latest acceptable departure date, YYYY-MM-DD"],
    preference: Annotated[
        Literal["cheapest", "fastest", "balanced"], "What the traveller cares about most"
    ] = "balanced",
) -> dict:
    """Analyse fares across a date window and recommend the best day and time slot to fly."""
    try:
        start, end = _parse_date(earliest_date), _parse_date(latest_date)
    except ValueError:
        return {"error": "Dates must be YYYY-MM-DD."}
    start = max(start, date.today())
    if end < start:
        return {"error": "latest_date must be on or after earliest_date (and not in the past)."}
    end = min(end, start + timedelta(days=45))  # keep the scan bounded

    def minutes(f: dict) -> int:
        h, m = f["duration"].replace("h", "").replace("m", "").split()
        return int(h) * 60 + int(m)

    def score(f: dict) -> float:
        if preference == "cheapest":
            return f["price_usd"]
        if preference == "fastest":
            return minutes(f) + f["stops"] * 120
        return f["price_usd"] + minutes(f) * 0.25 + f["stops"] * 40  # balanced

    daily, slot_prices, best = [], {}, None
    day = start
    while day <= end:
        flights = _flights_for_day(origin, destination, day)
        daily.append({"date": day.isoformat(), "weekday": day.strftime("%A"),
                      "lowest_fare_usd": flights[0]["price_usd"]})
        for f in flights:
            slot_prices.setdefault(f["time_slot"], []).append(f["price_usd"])
            if best is None or score(f) < score(best):
                best = f
        day += timedelta(days=1)

    daily.sort(key=lambda d: d["lowest_fare_usd"])
    slot_avg = sorted(
        ({"time_slot": s, "avg_fare_usd": round(sum(p) / len(p), 2)} for s, p in slot_prices.items()),
        key=lambda s: s["avg_fare_usd"],
    )
    days_ahead = (_parse_date(best["departure"][:10]) - date.today()).days
    tips = [
        f"Cheapest departure window on this route: {slot_avg[0]['time_slot']}.",
        "Tuesday/Wednesday departures are typically cheapest; Friday and Sunday are the most expensive.",
    ]
    if days_ahead < 21:
        tips.append("You are booking under 3 weeks out - fares tend to rise, so book soon.")
    else:
        tips.append("Booking 3-8 weeks ahead is usually the price sweet spot - good time to book.")

    return {
        "preference": preference,
        "recommended_flight": best,
        "cheapest_days": daily[:3],
        "most_expensive_day": daily[-1],
        "time_slots_by_avg_fare": slot_avg,
        "tips": tips,
    }


def search_nearby_stays(
    city: Annotated[str, "City to stay in (usually the flight destination)"],
    check_in: Annotated[str, "Check-in date YYYY-MM-DD"],
    check_out: Annotated[str, "Check-out date YYYY-MM-DD"],
    near: Annotated[
        Literal["airport", "city_center"], "Where the stay should be close to"
    ] = "airport",
    stay_type: Annotated[Literal["any", "hotel", "motel"], "Type of accommodation"] = "any",
    max_price_per_night: Annotated[float, "Maximum nightly price in USD (0 = no limit)"] = 0,
) -> dict:
    """Find hotels and motels near the airport or city centre, ranked by value (rating vs. price vs. distance)."""
    try:
        cin, cout = _parse_date(check_in), _parse_date(check_out)
    except ValueError:
        return {"error": "Dates must be YYYY-MM-DD."}
    nights = (cout - cin).days
    if nights <= 0:
        return {"error": "check_out must be after check_in."}

    rng = _rng("stays", city, near)
    hotel_names = ["Grand", "Skyline", "Plaza", "Regency", "Residency", "Marriott-style", "Park Inn", "Comfort Suites"]
    motel_names = ["Travelers Rest", "Runway Motor Inn", "Budget Stop", "Highway Lodge", "Airport Motel 8"]

    stays = []
    for kind, names, (lo, hi), stars in (("hotel", hotel_names, (70, 320), (3, 5)),
                                          ("motel", motel_names, (35, 95), (2, 3))):
        for name in rng.sample(names, k=min(len(names), rng.randint(3, 5))):
            rating = round(rng.uniform(3.2, 4.9 if kind == "hotel" else 4.4), 1)
            distance = round(rng.uniform(0.3, 6 if near == "airport" else 4), 1)
            price = round(rng.uniform(lo, hi), 2)
            stays.append({
                "name": f"{name} {city.title()}" if kind == "hotel" else f"{name} - {city.title()}",
                "type": kind,
                "stars": rng.randint(*stars),
                "guest_rating": rating,
                f"distance_to_{near}_km": distance,
                "price_per_night_usd": price,
                "total_price_usd": round(price * nights, 2),
                "amenities": rng.sample(
                    ["free wifi", "free breakfast", "airport shuttle", "parking", "pool", "gym",
                     "24h front desk", "late check-in"], k=4),
                "free_cancellation": rng.random() > 0.35,
            })

    if stay_type != "any":
        stays = [s for s in stays if s["type"] == stay_type]
    if max_price_per_night > 0:
        stays = [s for s in stays if s["price_per_night_usd"] <= max_price_per_night]
    for s in stays:  # higher is better
        s["value_score"] = round(s["guest_rating"] * 20 - s["price_per_night_usd"] * 0.15
                                 - s[f"distance_to_{near}_km"] * 3, 1)
    stays.sort(key=lambda s: s["value_score"], reverse=True)
    return {"city": city, "nights": nights, "count": len(stays), "stays": stays[:6]}


def book_flight(
    flight_id: Annotated[str, "flight_id returned by search_flights / recommend_best_travel_time"],
    passenger_name: Annotated[str, "Full passenger name as on passport/ID"],
    contact_email: Annotated[str, "Email for the e-ticket"],
) -> dict:
    """Book a flight. Only call after the user has explicitly confirmed the flight and passenger details."""
    flight = FLIGHT_INDEX.get(flight_id)
    if flight is None:
        return {"error": f"Unknown flight_id '{flight_id}'. Search for flights first."}
    if flight["seats_left"] < 1:
        return {"error": "Flight is sold out."}
    flight["seats_left"] -= 1
    pnr = uuid.uuid4().hex[:6].upper()
    BOOKINGS[pnr] = {"flight": flight, "passenger": passenger_name, "email": contact_email}
    return {"status": "CONFIRMED", "pnr": pnr, "flight": flight}
