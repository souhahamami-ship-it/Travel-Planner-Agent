"""Test script for the travel agent tools.

Usage (run from the project folder):
    python test_tools.py              # offline checks + live API checks
    python test_tools.py --offline    # offline checks only (no keys, no network)
    python test_tools.py --agent      # also run one end-to-end agent request
    python test_tools.py --verbose    # print full tool outputs and tracebacks

Offline checks never touch the network. Live checks call the real APIs and use a
little of your quota (about one request per tool). A live test is SKIPPED when its
API key is missing from .env.
"""

import argparse
import json
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import date, timedelta
from unittest.mock import patch

import requests
from dotenv import load_dotenv

load_dotenv()  # before importing Tools, like agent.py does

try:
    from Tools.tools import tools_list, tools_map
except ImportError as e:
    sys.exit(
        f"Could not import the tools: {e}\n"
        "Install dependencies: pip install langchain-core langchain-groq "
        "python-dotenv requests pydantic ddgs"
    )

VERBOSE = False
OFFLINE, LIVE, AGENT = [], [], []


# ---------------------------------------------------------------- helpers

class Skip(Exception):
    """Raised by a test that cannot run (for example a missing API key)."""


def offline(f):
    OFFLINE.append(f)
    return f


def live(f):
    LIVE.append(f)
    return f


def agent(f):
    AGENT.append(f)
    return f


def need_key(*names):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise Skip(f"missing {', '.join(missing)} in .env")


@contextmanager
def temp_env(name, value):
    """Temporarily set (or remove, if value is None) an environment variable."""
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def short(obj, limit=300):
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def call(tool_name, **args):
    result = tools_map[tool_name].invoke(args)
    if VERBOSE:
        print(f"    {tool_name}({args}) ->")
        print("    " + json.dumps(result, indent=2, ensure_ascii=False, default=str)[:1500])
    return result


def in_days(n):
    return (date.today() + timedelta(days=n)).isoformat()


def assert_shape(result):
    """Every tool must return {"ok": True, "data": ...} or {"ok": False, "error": "..."}."""
    assert isinstance(result, dict), f"expected a dict, got {type(result).__name__}"
    assert isinstance(result.get("ok"), bool), f"missing boolean 'ok': {short(result)}"
    if result["ok"]:
        assert "data" in result, f"ok=True but no 'data': {short(result)}"
    else:
        assert isinstance(result.get("error"), str) and result["error"], \
            f"ok=False but no 'error' text: {short(result)}"


def expect_error(result, *words):
    assert_shape(result)
    assert result["ok"] is False, f"expected ok=False, got {short(result)}"
    text = result["error"].lower()
    assert all(w.lower() in text for w in words), \
        f"error should mention {words}, got: {result['error']}"


def expect_ok(result):
    assert_shape(result)
    assert result["ok"] is True, f"expected ok=True, got {short(result)}"
    return result["data"]


# ------------------------------------------------------- offline: no network

@offline
def registry_has_all_tools():
    expected = {
        "convert_currency", "get_weather", "get_country_info",
        "search_flights", "search_travel_places", "suggest_activities",
    }
    assert set(tools_map) == expected, f"tools found: {sorted(tools_map)}"
    assert len(tools_list) == len(tools_map), "duplicate tool names in tools_list"
    for t in tools_list:
        assert t.description and len(t.description) > 40, f"{t.name}: description too short"
        assert t.args, f"{t.name}: no argument schema"
    return f"{len(tools_list)} tools registered"


@offline
def money_rejects_bad_input():
    expect_error(call("convert_currency", amount=10, from_currency="dollar", to_currency="EUR"), "iso")
    expect_error(call("convert_currency", amount=-5, from_currency="EUR", to_currency="USD"), "non-negative")


@offline
def weather_rejects_bad_dates():
    expect_error(
        call("get_weather", destination="Tunis", start_date="06/05/2026", end_date="06/07/2026"),
        "yyyy-mm-dd",
    )
    expect_error(
        call("get_weather", destination="Tunis", start_date=in_days(5), end_date=in_days(2)),
        "before",
    )


@offline
def flights_reject_bad_input():
    base = dict(origin="TUN", destination="CDG", departure_date=in_days(30))
    with temp_env("Search_API_KEY", "dummy"):
        expect_error(call("search_flights", **{**base, "origin": "Tunis"}), "iata")
        expect_error(call("search_flights", **{**base, "departure_date": in_days(-3)}), "past")
        expect_error(call("search_flights", **{**base, "departure_date": "30/11/2026"}), "yyyy-mm-dd")
        expect_error(call("search_flights", **{**base, "return_date": in_days(20)}), "before")
        expect_error(call("search_flights", **{**base, "passengers": 0}), "passengers")
        expect_error(call("search_flights", **{**base, "currency": "dollars"}), "currency")


@offline
def places_reject_bad_input():
    with temp_env("SERPAPI_KEY", "dummy"):
        expect_error(call("search_travel_places", city="  ", categories=["hotels"]), "city")
        expect_error(call("search_travel_places", city="Tunis", categories=[]), "category")


@offline
def destination_and_activities_reject_empty_input():
    with temp_env("RESTCOUNTRIES_API_KEY", "dummy"):
        expect_error(call("get_country_info", country_name="  "), "required")
    expect_error(call("suggest_activities", destination="  "), "required")


@offline
def tools_report_missing_api_keys():
    cases = [
        ("search_travel_places", "SERPAPI_KEY", dict(city="Tunis", categories=["hotels"])),
        ("search_flights", "Search_API_KEY",
         dict(origin="TUN", destination="CDG", departure_date=in_days(30))),
        ("get_country_info", "RESTCOUNTRIES_API_KEY", dict(country_name="Tunisia")),
    ]
    for tool_name, var, args in cases:
        with temp_env(var, None):
            expect_error(call(tool_name, **args), "not configured")


@offline
def errors_never_leak_api_keys():
    """Simulate a network failure whose message contains the key, like real requests errors do."""
    secret = "SECRET123"
    fake = requests.ConnectionError(
        f"Max retries exceeded with url: /search.json?q=x&api_key={secret}"
    )
    cases = [
        ("search_travel_places", "SERPAPI_KEY", dict(city="Tunis", categories=["hotels"])),
        ("search_flights", "Search_API_KEY",
         dict(origin="TUN", destination="CDG", departure_date=in_days(30))),
        ("get_country_info", "RESTCOUNTRIES_API_KEY", dict(country_name="Tunisia")),
    ]
    for tool_name, var, args in cases:
        with temp_env(var, secret), patch("requests.get", side_effect=fake):
            result = call(tool_name, **args)
        assert_shape(result)
        assert secret not in json.dumps(result), f"{tool_name} leaked the API key: {short(result)}"


# ------------------------------------------------------ live: real API calls

@live
def money_live_conversion():
    d = expect_ok(call("convert_currency", amount=100, from_currency="EUR", to_currency="USD"))
    assert d["converted"] > 0 and d["rate"] > 0, short(d)
    return f"100 EUR = {d['converted']} USD (rate {d['rate']}, updated {d.get('rates_updated')})"


@live
def money_live_unsupported_code():
    r = call("convert_currency", amount=1, from_currency="XXX", to_currency="USD")
    expect_error(r)
    return f"error text: {r['error']}"


@live
def weather_live_forecast():
    d = expect_ok(call("get_weather", destination="Tunis", start_date=in_days(2), end_date=in_days(4)))
    assert len(d["days"]) == 3, f"expected 3 days, got {len(d['days'])}"
    assert d["days"][0]["high_c"] is not None, short(d["days"][0])
    return f"{d['location']}: first day {d['days'][0]}"


@live
def weather_live_far_future_is_an_error():
    r = call("get_weather", destination="Tunis", start_date=in_days(200), end_date=in_days(203))
    expect_error(r)
    return f"error text: {r['error']}"


@live
def destination_live_exact_country():
    need_key("RESTCOUNTRIES_API_KEY")
    d = expect_ok(call("get_country_info", country_name="Tunisia"))
    assert d["name"] == "Tunisia", short(d)
    assert any("tunis" in c.lower() for c in d["capitals"]), f"capitals: {d['capitals']}"
    return (f"capitals={d['capitals']} currencies={d['currencies']} "
            f"languages={d['languages']} population={d['population']}")


@live
def destination_live_picks_right_country():
    need_key("RESTCOUNTRIES_API_KEY")
    d = expect_ok(call("get_country_info", country_name="India"))
    assert d["name"] == "India", f"got {d['name']} instead of India"


@live
def destination_live_typo_gives_suggestion():
    need_key("RESTCOUNTRIES_API_KEY")
    r = call("get_country_info", country_name="Tunisa")
    expect_error(r)
    return f"error text: {r['error']}"


@live
def places_live_search():
    need_key("SERPAPI_KEY")
    d = expect_ok(call("search_travel_places", city="Tunis", categories=["museums"], limit=3))
    assert len(d) >= 1 and d[0]["name"], f"no places returned: {short(d)}"
    return f"{len(d)} places, first: {d[0]['name']} (rating {d[0]['rating']})"


@live
def flights_live_search():
    need_key("Search_API_KEY")
    d = expect_ok(call("search_flights", origin="TUN", destination="CDG", departure_date=in_days(30)))
    assert d["offers"], f"no offers returned: {short(d)}"
    first = d["offers"][0]
    return f"{len(d['offers'])} offers, first: {first['price']} {first['currency']}, {first['stops']} stop(s)"


@live
def activities_live_search():
    d = expect_ok(call("suggest_activities", destination="Tunis", interests="history", max_results=3))
    assert d["results"], f"no results (DuckDuckGo may be rate-limiting): {short(d)}"
    return f"{len(d['results'])} results, first: {d['results'][0]['title']}"


# ------------------------------------------- agent: end to end (uses Groq key)

@agent
def agent_end_to_end():
    need_key("GROQ_API_KEY")
    from agent import new_conversation, run_agent

    messages = new_conversation()
    answer = run_agent(
        "Convert 100 EUR to TND, and give me the weather forecast for Tunis for the next 3 days.",
        messages,
    )
    used = [tc["name"] for m in messages if getattr(m, "tool_calls", None) for tc in m.tool_calls]

    assert isinstance(answer, str) and answer.strip(), "agent returned an empty answer"
    assert "step limit" not in answer.lower(), "agent hit the step limit"
    assert "convert_currency" in used, f"convert_currency was never called (tools used: {used})"
    return f"tools used: {used}\n    answer: {answer[:400]}"


# ------------------------------------------------------------------ runner

def run_group(title, tests):
    print(f"\n=== {title} ===")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for test in tests:
        try:
            detail, status = test() or "", "PASS"
        except Skip as s:
            detail, status = str(s), "SKIP"
        except AssertionError as e:
            detail, status = str(e), "FAIL"
        except Exception as e:
            detail, status = f"{type(e).__name__}: {e}", "FAIL"
            if VERBOSE:
                detail += "\n" + traceback.format_exc()
        counts[status] += 1
        if not VERBOSE and len(detail) > 500:
            detail = detail[:500] + "..."
        print(f"[{status}] {test.__name__}" + (f" - {detail}" if detail else ""))
    return counts


def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="Test the travel agent tools.")
    parser.add_argument("--offline", action="store_true", help="run offline checks only")
    parser.add_argument("--agent", action="store_true", help="also run the end-to-end agent test")
    parser.add_argument("--verbose", action="store_true", help="print full outputs and tracebacks")
    args = parser.parse_args()
    VERBOSE = args.verbose

    totals = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    groups = [("Offline checks (no network)", OFFLINE)]
    if not args.offline:
        groups.append(("Live API checks", LIVE))
        if args.agent:
            groups.append(("End-to-end agent", AGENT))

    for title, tests in groups:
        for status, n in run_group(title, tests).items():
            totals[status] += n

    print(f"\nSummary: {totals['PASS']} passed, {totals['FAIL']} failed, {totals['SKIP']} skipped")
    sys.exit(1 if totals["FAIL"] else 0)


if __name__ == "__main__":
    main()