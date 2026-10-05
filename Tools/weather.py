from datetime import date

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10  # seconds


class WeatherInput(BaseModel):
    destination: str = Field(
        description="City name only, e.g. 'Tunis' or 'Paris'. Do not add the country here."
    )
    start_date: str = Field(description="First day, format YYYY-MM-DD.")
    end_date: str = Field(
        description="Last day, format YYYY-MM-DD. Must not be before start_date."
    )
    country_code: str = Field(
        default="",
        description="Optional 2-letter ISO country code (e.g. 'TN', 'FR') to pick the right city when several share a name.",
    )


@tool(args_schema=WeatherInput)
def get_weather(
    destination: str,
    start_date: str,
    end_date: str,
    country_code: str = "",
) -> dict:
    """Get a daily weather forecast (high and low temperature in °C, and rain
    probability in %) for a destination between two dates.
    Forecasts only cover roughly the next 16 days. For trips further away this
    tool returns an error: then give general seasonal advice and say clearly
    that it is not a live forecast."""

    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError:
        return {"ok": False, "error": "Dates must use the format YYYY-MM-DD."}

    if end < start:
        return {"ok": False, "error": "end_date must not be before start_date."}

    # The geocoder matches the city name only, so "Tunis, Tunisia" would fail.
    name = destination.split(",")[0].strip()
    if not name:
        return {"ok": False, "error": "destination is required."}

    try:
        location = _geocode(name, country_code)
        if location is None:
            return {"ok": False, "error": f"Could not find destination: {destination}"}
        days = _forecast(location["latitude"], location["longitude"], start, end)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    except (requests.RequestException, ValueError) as e:
        return {"ok": False, "error": f"Weather request failed ({type(e).__name__})."}

    return {
        "ok": True,
        "data": {
            # Show which place was actually matched, in case it's not the one meant.
            "location": ", ".join(
                p for p in (location.get("name"), location.get("country")) if p
            ),
            "days": days,
        },
    }


def _geocode(name: str, country_code: str) -> dict | None:
    params = {"name": name, "count": 1, "language": "en", "format": "json"}
    if country_code.strip():
        params["countryCode"] = country_code.strip().upper()

    response = requests.get(GEOCODING_URL, params=params, timeout=TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"Geocoding service returned HTTP {response.status_code}.")

    results = response.json().get("results")
    return results[0] if results else None


def _forecast(latitude: float, longitude: float, start: date, end: date) -> list[dict]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    response = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT)
    data = response.json()

    # Open-Meteo reports problems (e.g. dates out of range) as {"error": true, "reason": "..."}
    if response.status_code != 200 or data.get("error"):
        reason = data.get("reason", f"HTTP {response.status_code}")
        raise RuntimeError(f"Forecast unavailable: {reason}")

    daily = data.get("daily") or {}
    times = daily.get("time", [])
    highs = daily.get("temperature_2m_max") or [None] * len(times)
    lows = daily.get("temperature_2m_min") or [None] * len(times)
    rain = daily.get("precipitation_probability_max") or [None] * len(times)

    return [
        {
            "date": d,
            "high_c": hi,
            "low_c": lo,
            "rain_probability_percent": rp,  # can be None if not available
        }
        for d, hi, lo, rp in zip(times, highs, lows, rain)
    ]