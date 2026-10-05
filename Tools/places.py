import os

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

SERPAPI_URL = "https://serpapi.com/search.json"
REQUEST_TIMEOUT = 15   # seconds, per request
MAX_CATEGORIES = 5     # cap sequential requests per tool call
DEFAULT_LIMIT = 5
MAX_LIMIT = 10

# Google Maps understands plain words like "hotels" or "cafe", so only
# terms that need rewording are listed here.
CATEGORY_ALIASES = {
    "attraction": "tourist attractions",
    "attractions": "tourist attractions",
}


class SearchPlacesInput(BaseModel):
    city: str = Field(description="City to search in, e.g. 'Tunis' or 'Paris'.")
    categories: list[str] = Field(
        description="1-5 place types to search, e.g. ['hotels', 'restaurants', 'museums', 'beaches']."
    )
    country: str = Field(
        default="",
        description="Country of the city, to avoid ambiguity (e.g. 'France'). Optional.",
    )
    limit: int = Field(
        default=DEFAULT_LIMIT,
        description=f"Maximum results per category (1-{MAX_LIMIT}).",
    )


@tool(args_schema=SearchPlacesInput)
def search_travel_places(
    city: str,
    categories: list[str],
    country: str = "",
    limit: int = DEFAULT_LIMIT,
) -> dict:
    """Search real places in a city using Google Maps: hotels, hostels, restaurants,
    cafes, bars, museums, beaches, parks, markets, tourist attractions, and so on.
    Returns name, address, phone, website, rating, review count, price level and a
    Maps link for each place. Use it for where to stay, eat, or visit.
    Do not use it for flights, weather, or currency."""

    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return {"ok": False, "error": "SERPAPI_KEY is not configured."}

    city = city.strip()
    if not city:
        return {"ok": False, "error": "city is required."}

    cleaned = [c.strip().lower() for c in categories if c and c.strip()]
    if not cleaned:
        return {"ok": False, "error": "At least one category is required."}

    cleaned = cleaned[:MAX_CATEGORIES]
    limit = max(1, min(limit, MAX_LIMIT))
    location = f"{city}, {country.strip()}" if country.strip() else city

    places: list[dict] = []
    errors: list[str] = []

    for category in cleaned:
        search_term = CATEGORY_ALIASES.get(category, category)
        query = f"{search_term} in {location}"

        try:
            places.extend(_fetch_category(api_key, query, search_term, limit))
        except RuntimeError as e:
            # Our own messages, safe to show the model.
            errors.append(f"{search_term}: {e}")
        except (requests.RequestException, ValueError) as e:
            # Never include str(e): requests errors contain the full URL,
            # which includes the API key.
            errors.append(f"{search_term}: request failed ({type(e).__name__})")

    places = remove_duplicates(places)

    if not places and errors:
        return {"ok": False, "error": "; ".join(errors)}

    result = {"ok": True, "data": places}
    if errors:
        result["warnings"] = errors  # some categories failed, others worked
    return result


def _fetch_category(api_key: str, query: str, category: str, limit: int) -> list[dict]:
    params = {
        "engine": "google_maps",
        "q": query,
        "type": "search",
        "hl": "en",
        "api_key": api_key,
    }
    response = requests.get(SERPAPI_URL, params=params, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        raise RuntimeError(f"SerpAPI returned HTTP {response.status_code}")

    data = response.json()

    # SerpAPI can answer HTTP 200 with an "error" field.
    error = data.get("error")
    if error:
        if "hasn't returned any results" in error:
            return []
        raise RuntimeError(error)

    results = data.get("local_results")
    if not results and data.get("place_results"):
        # A single exact match comes back as place_results instead.
        results = [data["place_results"]]

    return [_clean_place(p, category) for p in (results or [])[:limit]]


def _clean_place(place: dict, category: str) -> dict:
    """Keep only what the model needs to write an answer (saves tokens)."""
    return {
        "name": place.get("title") or "",
        "category": category,
        "address": place.get("address") or "",
        "phone": place.get("phone") or "",
        "website": place.get("website") or "",
        "rating": place.get("rating"),
        "reviews": place.get("reviews"),
        "price": place.get("price") or "",
        "maps_url": place.get("link") or "",
    }


def remove_duplicates(places: list[dict]) -> list[dict]:
    unique = {}
    for place in places:
        key = (place["name"].lower(), place["address"].lower())
        unique.setdefault(key, place)
    return list(unique.values())