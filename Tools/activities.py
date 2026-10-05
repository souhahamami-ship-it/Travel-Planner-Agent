from ddgs import DDGS
from langchain_core.tools import tool
from pydantic import BaseModel, Field

SEARCH_TIMEOUT = 10  # seconds
DEFAULT_RESULTS = 5
MAX_RESULTS = 8
MAX_SUMMARY_CHARS = 300


class ActivitiesInput(BaseModel):
    destination: str = Field(description="City or region, e.g. 'Tunis' or 'Kyoto'.")
    interests: str = Field(
        default="",
        description="Optional interests to focus on, e.g. 'food, history, hiking'.",
    )
    max_results: int = Field(
        default=DEFAULT_RESULTS,
        description=f"Number of web results to return (1-{MAX_RESULTS}).",
    )


@tool(args_schema=ActivitiesInput)
def suggest_activities(
    destination: str,
    interests: str = "",
    max_results: int = DEFAULT_RESULTS,
) -> dict:
    """Search the web for ideas of things to do in a destination (sights,
    experiences, tours), optionally focused on interests. Returns web search
    results (title, short summary, link), not verified recommendations: present
    them as search results and mention the source. For specific venues with
    ratings, addresses and prices (hotels, restaurants, museums), use
    search_travel_places instead."""

    destination = destination.strip()
    if not destination:
        return {"ok": False, "error": "destination is required."}

    max_results = max(1, min(max_results, MAX_RESULTS))
    query = f"best things to do in {destination} {interests.strip()}".strip()

    try:
        raw_results = DDGS(timeout=SEARCH_TIMEOUT).text(query, max_results=max_results)
    except Exception as e:
        # ddgs can raise several different exception types (rate limit, timeout,
        # no results, ...), so a broad catch is reasonable here.
        if "no results" in str(e).lower():
            raw_results = []
        else:
            return {"ok": False, "error": f"Activity search failed ({type(e).__name__})."}

    results = [
        {
            "title": r.get("title") or "",
            "summary": (r.get("body") or "")[:MAX_SUMMARY_CHARS],
            "url": r.get("href") or "",
        }
        for r in raw_results or []
    ]

    data = {"query": query, "results": results}
    if not results:
        data["note"] = f"No web results found for {destination}."

    return {"ok": True, "data": data}