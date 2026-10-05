import math
import time

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field

API_URL = "https://open.er-api.com/v6/latest/{base}"
TIMEOUT = 10        # seconds
CACHE_TTL = 3600    # rates change about once a day, so 1 hour is fresh enough

# base currency -> (time fetched, full API response)
_cache: dict[str, tuple[float, dict]] = {}


class ConvertInput(BaseModel):
    amount: float = Field(description="Amount of money to convert, e.g. 250.")
    from_currency: str = Field(
        description="3-letter ISO code of the currency to convert from, e.g. 'EUR', 'USD', 'TND'."
    )
    to_currency: str = Field(
        description="3-letter ISO code of the currency to convert to, e.g. 'TND'."
    )


@tool(args_schema=ConvertInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between two currencies using exchange rates that are
    updated about once per day. The result is approximate, not a live trading
    rate, and real bank or exchange-office rates will differ.
    Always pass 3-letter ISO codes such as EUR, USD, TND, not symbols or names."""

    from_curr = from_currency.strip().upper()
    to_curr = to_currency.strip().upper()

    for code in (from_curr, to_curr):
        if not (len(code) == 3 and code.isalpha()):
            return {
                "ok": False,
                "error": f"'{code}' is not a 3-letter ISO currency code (e.g. EUR, USD, TND).",
            }

    if not math.isfinite(amount) or amount < 0:
        return {"ok": False, "error": "amount must be a non-negative number."}

    try:
        table = _get_rates(from_curr)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    except (requests.RequestException, ValueError) as e:
        return {"ok": False, "error": f"Exchange rate request failed ({type(e).__name__})."}

    rate = table["rates"].get(to_curr)
    if rate is None:
        return {"ok": False, "error": f"No exchange rate available for {to_curr}."}

    return {
        "ok": True,
        "data": {
            "amount": amount,
            "from": from_curr,
            "to": to_curr,
            "converted": round(amount * rate, 2),
            "rate": rate,  # 1 unit of `from` in `to`, unrounded
            "rates_updated": table.get("time_last_update_utc"),
        },
    }


def _get_rates(base: str) -> dict:
    cached = _cache.get(base)
    if cached and time.time() - cached[0] < CACHE_TTL:
        return cached[1]

    response = requests.get(API_URL.format(base=base), timeout=TIMEOUT)
    data = response.json()

    if data.get("result") != "success":
        if data.get("error-type") == "unsupported-code":
            raise RuntimeError(f"Unsupported currency code: {base}.")
        raise RuntimeError("Exchange rate service returned an error.")

    _cache[base] = (time.time(), data)
    return data