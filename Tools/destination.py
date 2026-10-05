import difflib
import os
from urllib.parse import quote

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

BASE_URL = "https://api.restcountries.com/countries/v5"
API_KEY_ENV = "RESTCOUNTRIES_API_KEY"  # get a free key at restcountries.com/sign-up
DEMO_KEY = "rc_live_demo"              # public test key: returns samples only
TIMEOUT = 10

# Subfield names inside `languages` / `currencies` objects. The API docs describe
# them only loosely, so we try several likely names. Print one real response once
# and trim this list to what you actually see.
LABEL_KEYS = ("name", "english_name", "name_english", "english", "code")

STATUS_MESSAGES = {
    401: "Country service rejected the API key (check RESTCOUNTRIES_API_KEY).",
    403: "Country service denied access (monthly request limit reached, or account frozen).",
    429: "Country service rate limit reached; try again in a few seconds.",
}

_country_names: list[str] = []  # cache used only for "did you mean" suggestions


class CountryInput(BaseModel):
    country_name: str = Field(
        description="Country name in English, e.g. 'Tunisia' or 'Japan'."
    )


@tool(args_schema=CountryInput)
def get_country_info(country_name: str) -> dict:
    """Get background facts about a destination country: official name, capital(s),
    region, population, languages, currencies (with ISO codes), calling code, time
    zones and driving side. Use it for country background only, not for live
    prices, weather or flights."""

    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        return {"ok": False, "error": f"{API_KEY_ENV} is not configured."}

    name = country_name.strip()
    if not name:
        return {"ok": False, "error": "country_name is required."}

    # The key goes in a header, not the URL, so it can't appear in request error messages.
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        country, meta = _find_country(name, headers)
    except LookupError as e:
        return {"ok": False, "error": str(e)}
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    except (requests.RequestException, ValueError) as e:
        return {"ok": False, "error": f"Country request failed ({type(e).__name__})."}

    result = {"ok": True, "data": _summarize(country)}
    if meta.get("_demo"):
        result["warnings"] = ["Demo API key in use: the data is a sample, not guaranteed for this country."]
    return result


def _get(path: str, headers: dict, params: dict | None = None) -> tuple[list[dict], dict]:
    """Call the API. Returns (objects, data). A 404 means 'no match' -> empty list."""
    response = requests.get(f"{BASE_URL}{path}", headers=headers, params=params, timeout=TIMEOUT)

    if response.status_code == 404:
        return [], {}
    if response.status_code in STATUS_MESSAGES:
        raise RuntimeError(STATUS_MESSAGES[response.status_code])
    if response.status_code != 200:
        raise RuntimeError(f"Country service returned HTTP {response.status_code}.")

    data = response.json().get("data") or {}  # v5 wraps results as {"data": {"objects": [...]}}
    return data.get("objects") or [], data


def _find_country(name: str, headers: dict) -> tuple[dict, dict]:
    # 1. Exact (case-insensitive) match on the common name, e.g. "Tunisia".
    objects, data = _get(f"/names.common/{quote(name)}", headers)
    if objects:
        return objects[0], data

    # 2. Substring search across all name fields. Results come back alphabetically,
    #    so never blindly take the first one ("India" would hit "British Indian Ocean Territory").
    objects, data = _get("/name", headers, params={"q": name, "limit": 10})

    exact = [c for c in objects if name.lower() in _all_names(c)]
    if exact:
        return exact[0], data
    if len(objects) == 1:
        return objects[0], data
    if objects:
        options = ", ".join(c["names"]["common"] for c in objects if (c.get("names") or {}).get("common"))
        raise LookupError(f"Several countries match '{name}': {options}. Use the exact country name.")

    suggestions = _suggest(name, headers)
    message = f"Country not found: '{name}'."
    if suggestions:
        message += f" Did you mean: {', '.join(suggestions)}?"
    raise LookupError(message)


def _all_names(country: dict) -> set[str]:
    names = country.get("names") or {}
    values = [names.get("common"), names.get("official"), *(names.get("alternates") or [])]
    return {v.lower() for v in values if isinstance(v, str)}


def _suggest(name: str, headers: dict) -> list[str]:
    """Spelling suggestions for the error message (we never auto-correct silently)."""
    try:
        if not _country_names:
            offset = 0
            while offset < 300:  # ~250 countries, 100 per page on the free plan
                objects, data = _get(
                    "", headers,
                    params={"response_fields": "names.common", "limit": 100, "offset": offset},
                )
                _country_names.extend(
                    o["names"]["common"] for o in objects if (o.get("names") or {}).get("common")
                )
                if not (data.get("meta") or {}).get("more"):
                    break
                offset += 100
    except (RuntimeError, requests.RequestException, ValueError):
        _country_names.clear()
        return []

    lookup = {n.lower(): n for n in _country_names}
    return [lookup[m] for m in difflib.get_close_matches(name.lower(), lookup, n=3, cutoff=0.6)]


def _summarize(country: dict) -> dict:
    """Keep only what the model needs (the raw record has 90+ fields)."""
    names = country.get("names") or {}
    capitals = [
        c["name"] for c in country.get("capitals") or []
        if isinstance(c, dict) and c.get("name")
    ]
    return {
        "name": names.get("common"),
        "official_name": names.get("official"),
        "capitals": capitals,
        "region": country.get("region"),
        "subregion": country.get("subregion"),
        "population": country.get("population"),
        "languages": _labels(country.get("languages")),
        "currencies": _labels(country.get("currencies")),
        "calling_codes": country.get("calling_codes") or [],
        "timezones": country.get("timezones") or [],
        "driving_side": (country.get("cars") or {}).get("driving_side"),
    }


def _labels(value) -> list[str]:
    """Turn a list or dict of strings/objects into readable labels,
    e.g. {"TND": {"name": "Tunisian dinar"}} -> ["Tunisian dinar (TND)"]."""
    if not value:
        return []

    entries = value.items() if isinstance(value, dict) else [(None, v) for v in value]
    labels = []
    for key, item in entries:
        if isinstance(item, dict):
            label = next((str(item[k]) for k in LABEL_KEYS if item.get(k)), None)
            if label and key:
                label = f"{label} ({key})"
            label = label or key
        else:
            label = item if isinstance(item, str) else key
        if label:
            labels.append(str(label))
    return labels