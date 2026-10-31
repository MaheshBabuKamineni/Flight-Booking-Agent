"""Offline smoke test for the travel tools (no API key needed): python test_tools.py"""

from datetime import date, timedelta

from travel_tools import book_flight, recommend_best_travel_time, search_flights, search_nearby_stays

d1 = (date.today() + timedelta(days=10)).isoformat()
d2 = (date.today() + timedelta(days=20)).isoformat()

flights = search_flights("Hyderabad", "Dubai", d1)
assert flights["count"] > 0
print("cheapest flight:", flights["flights"][0])

best = recommend_best_travel_time("Hyderabad", "Dubai", d1, d2)
print("recommended:", best["recommended_flight"]["flight_id"], "| cheapest slot:", best["time_slots_by_avg_fare"][0])
print("tips:", best["tips"])

stays = search_nearby_stays("Dubai", d1, d2, max_price_per_night=150)
assert all(s["price_per_night_usd"] <= 150 for s in stays["stays"])
print("best-value stay:", stays["stays"][0]["name"], stays["stays"][0]["type"])

booking = book_flight(flights["flights"][0]["flight_id"], "Test User", "test@example.com")
assert booking["status"] == "CONFIRMED"
assert "passenger" not in booking and "email" not in booking
print("booking:", booking["pnr"])
assert "error" in book_flight("NOPE-1", "x", "y")
print("All tool checks passed.")
