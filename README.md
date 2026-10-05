# Travel Planner Agent

A command-line AI travel assistant built to learn how tool-calling agents work. You describe a trip in plain language, and the agent decides which tools to call (flights, weather, places, currency, country facts, activity ideas), runs them, and writes a travel plan from the results.

Built with **LangChain** and **Groq**. The agent loop is written by hand, without a framework, so every step is visible.

```
You: Plan 4 days in Rome from Tunis in two weeks, budget 800 EUR
[tool] get_weather({'destination': 'Rome', 'start_date': '2026-10-19', ...})
[tool] search_flights({'origin': 'TUN', 'destination': 'FCO', ...})
[tool] search_travel_places({'city': 'Rome', 'categories': ['hotels', 'hostels'], ...})
[tool] suggest_activities({'destination': 'Rome', ...})

Travel Assistant:
...a day-by-day plan with flights, hotels, weather and a budget breakdown...
```

## Features

- Plans trips by combining several live data sources in one answer
- Keeps conversation history, so you can follow up ("make it cheaper")
- Prints every tool call so you can watch the agent's decisions
- Tools return a consistent `{"ok": ..., "data" / "error": ...}` shape, so the model can tell success from failure
- Safe error handling: API keys never appear in error messages
- A test script with offline checks, live API checks, and an end-to-end agent test

## Project structure

```
travel_agent/
├── main.py            # Command-line chat loop
├── agent.py           # Model setup, tool execution, the agent loop
├── prompts.py         # System prompt (rules for the model, today's date)
├── test_tools.py      # Offline + live + end-to-end tests
├── .env               # Your API keys (never commit this)
└── Tools/
    ├── tools.py       # Registry: tools_list and tools_map
    ├── flights.py     # search_flights
    ├── weather.py     # get_weather
    ├── places.py      # search_travel_places
    ├── activities.py  # suggest_activities
    ├── destination.py # get_country_info
    └── money.py       # convert_currency
```

## Tools

| Tool | What it does | Data source | Key needed |
|---|---|---|---|
| `search_flights` | Flight offers with price, airlines, stops, duration | SearchApi.io (Google Flights) | `Search_API_KEY` |
| `get_weather` | Daily high/low and rain probability | Open-Meteo | none |
| `search_travel_places` | Hotels, restaurants, museums, etc. with ratings and addresses | SerpAPI (Google Maps) | `SERPAPI_KEY` |
| `suggest_activities` | Web search results with ideas for things to do | DuckDuckGo (`ddgs`) | none |
| `get_country_info` | Capital, languages, currencies, time zones, driving side | REST Countries v5 | `RESTCOUNTRIES_API_KEY` |
| `convert_currency` | Currency conversion with daily rates | open.er-api.com | none |

## Setup

Requires **Python 3.10+**.

```bash
pip install langchain-core langchain-groq python-dotenv requests pydantic ddgs
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_groq_key
SERPAPI_KEY=your_serpapi_key
Search_API_KEY=your_searchapi_io_key
RESTCOUNTRIES_API_KEY=your_restcountries_key

# optional: override the default model (qwen/qwen3.8-27b)
GROQ_MODEL=
```

## Usage

```bash
python main.py
```

Type a request and press Enter. Type `exit` to quit. The agent remembers the conversation, so follow-up questions work.

If you leave out important details (origin, dates, budget), the agent is instructed to ask instead of guessing.

## How it works

1. `main.py` reads your message and passes it to `run_agent()`.
2. `agent.py` sends the conversation to the model with the six tools attached.
3. If the model answers with tool calls, each one is executed. The result is added to the conversation as a `ToolMessage`.
4. The model is called again with the results. Steps 3–4 repeat until it answers without tool calls, up to `MAX_STEPS` (8) per request.
5. The final text is printed.

`prompts.py` holds the system prompt. It tells the model to use tools for live data, never invent data from failed tools, ask for missing details, and separate retrieved data from general advice. Today's date is added automatically so the model can resolve phrases like "in two weeks".

## Testing

```bash
python test.py              # offline + live checks
python test.py --offline    # offline only (no keys, no network)
python test.py --agent      # also run one end-to-end agent request
python test.py --verbose    # full outputs and tracebacks
```

- **Offline checks** verify the tool registry, input validation, missing-key handling, and that errors never leak API keys.
- **Live checks** call each real API once. A test is skipped if its key is missing.
- **Agent test** sends a real request through the model and checks that it calls the expected tool.

Live tests use a small amount of your API quota.

## Design notes

- **Tool results are structured.** Every tool returns `{"ok": True, "data": ...}` or `{"ok": False, "error": "..."}`, with an optional `"warnings"` field for partial failures.
- **Tool descriptions matter.** Each tool has a schema with a described field for every argument. The model reads these to decide when and how to call a tool.
- **Keys stay out of errors.** `requests` error messages contain the full URL, which includes query-string API keys. Tools report only the exception type or HTTP status.
- **Outputs are trimmed.** Tools return only the fields the model needs, and `agent.py` caps each tool result at 4000 characters.

## Known limitations

- **Weather** forecasts cover only about the next 16 days. For later dates the tool returns an error and the model should give general seasonal advice.
- **Flights:** for round trips, the price covers the whole trip but only outbound flights are listed. Airports must be given as IATA codes (the model usually knows the common ones).
- **Hotel prices** from Google Maps are indicative, not date-specific, and may be in USD.
- **Activity ideas** come from web search snippets and are not verified recommendations. DuckDuckGo can rate-limit or vary between runs.
- **The model can make mistakes** in arithmetic, itinerary logic, and mixing its own knowledge with tool data. Check the totals and plans.
- Tools run one after another, not in parallel.

## Ideas for next steps

- Tighten the system prompt: count nights correctly, always use `convert_currency`, label estimated costs, and never adjust estimates to fit a budget
- Convert place prices to the user's currency inside the tool
- Add an airport-code lookup tool for city names
- Run independent tool calls in parallel
- Save plans to a file, or add a simple web interface
- Try LangGraph once the manual loop feels familiar