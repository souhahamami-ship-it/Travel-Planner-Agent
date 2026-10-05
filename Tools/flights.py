import os
from datetime import date

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

API_URL = "https://www.searchapi.io/api/v1/search"
API_KEY_ENV = "Search_API_KEY"  # name of the variable in your .env file
TIMEOUT = 20
MAX_RESULTS = 5
MAX_PASSENGERS = 9


class FlightSearchInput(BaseModel):
    origin: str = Field(
        description="Departure airport 3-letter IATA code, e.g. 'TUN' (Tunis), 'CDG' (Paris)."
    )
    destination: str = Field(
        description="Arrival airport 3-letter IATA code, e.g. 'FCO' (Rome)."
    )
    departure_date: str = Field(
        description="Outbound date, format YYYY-MM-DD. Must be today or later."
    )
    return_date: str | None = Field(
        default=None,
        description="Return date, format YYYY-MM-DD, for a round trip. Omit for one-way.",
    )
    passengers: int = Field(
        default=1, description=f"Number of adult passengers (1-{MAX_PASSENGERS})."
    )
    currency: str = Field(
        default="USD", description="3-letter currency code for prices, e.g. 'USD', 'EUR'."
    )


@tool(args_schema=FlightSearchInput)
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    passengers: int = 1,
    currency: str = "USD",
) -> dict:
    """Search flight offers on Google Flights. Needs airport IATA codes, not city
    names. Returns up to 5 options with price, airlines, flight numbers, times,
    stops and duration. For round trips the price covers the whole round trip but
    only the outbound flights are listed. Prices are indicative and can change."""

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        return {"ok": False, "error": f"{API_KEY_ENV} is not configured."}

    origin = origin.strip().upper()
    destination = destination.strip().upper()
    currency = currency.strip().upper()

    for label, code in (("origin", origin), ("destination", destination)):
        if not (len(code) == 3 and code.isalpha()):
            return {
                "ok": False,
                "error": f"{label} must be a 3-letter IATA airport code (e.g. TUN, CDG), got '{code}'.",
            }
    if not (len(currency) == 3 and currency.isalpha()):
        return {"ok": False, "error": "currency must be a 3-letter code such as USD or EUR."}

    try:
        departure = date.fromisoformat(departure_date)
        returning = date.fromisoformat(return_date) if return_date else None
    except ValueError:
        return {"ok": False, "error": "Dates must use the format YYYY-MM-DD."}

    if departure < date.today():
        return {"ok": False, "error": "departure_date is in the past."}
    if returning and returning < departure:
        return {"ok": False, "error": "return_date must not be before departure_date."}
    if not 1 <= passengers <= MAX_PASSENGERS:
        return {"ok": False, "error": f"passengers must be between 1 and {MAX_PASSENGERS}."}

    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": departure.isoformat(),
        "adults": passengers,
        "currency": currency,
        "flight_type": "round_trip" if returning else "one_way",
        "api_key": api_key,
    }
    if returning:
        params["return_date"] = returning.isoformat()

    try:
        offers = _fetch_offers(params)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    except (requests.RequestException, ValueError) as e:
        # Never include str(e): request errors contain the URL, and the URL contains the API key.
        return {"ok": False, "error": f"Flight search request failed ({type(e).__name__})."}

    data = {
        "route": f"{origin} -> {destination}",
        "trip_type": params["flight_type"],
        "offers": [_parse_offer(o, currency) for o in offers[:MAX_RESULTS]],
    }
    if not offers:
        data["note"] = "No flights found for this route and date."
    elif returning:
        data["note"] = "Prices are for the round trip; only outbound flights are listed."

    return {"ok": True, "data": data}


def _fetch_offers(params: dict) -> list[dict]:
    response = requests.get(API_URL, params=params, timeout=TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"Flight search service returned HTTP {response.status_code}.")

    data = response.json()
    if data.get("error"):
        raise RuntimeError(f"Flight search error: {data['error']}")

    # Keep the API's own ordering: top options first, then the rest.
    return (data.get("best_flights") or []) + (data.get("other_flights") or [])


def _parse_offer(offer: dict, currency: str) -> dict:
    legs = []
    for leg in offer.get("flights") or []:
        departure = leg.get("departure_airport") or {}
        arrival = leg.get("arrival_airport") or {}
        legs.append(
            {
                "airline": leg.get("airline") or "Unknown airline",
                "flight_number": leg.get("flight_number") or "",
                "from": departure.get("id") or departure.get("name") or "",
                "departs": departure.get("time") or "",
                "to": arrival.get("id") or arrival.get("name") or "",
                "arrives": arrival.get("time") or "",
            }
        )

    return {
        "price": offer.get("price"),
        "currency": currency,
        "total_duration_min": offer.get("total_duration"),
        "stops": max(len(legs) - 1, 0),
        "legs": legs,
    }