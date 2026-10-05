"""Single place where the agent's tools are registered.

To add a new tool: create it in its own file under Tools/, import it here,
and add it to tools_list. agent.py picks it up automatically.
"""

from Tools.activities import suggest_activities
from Tools.destination import get_country_info
from Tools.flights import search_flights
from Tools.money import convert_currency
from Tools.places import search_travel_places
from Tools.weather import get_weather

tools_list = [
    convert_currency,
    get_weather,
    get_country_info,
    search_flights,
    search_travel_places,
    suggest_activities,
]

# name -> tool, used by the agent loop to run the tool the model asked for
tools_map = {tool.name: tool for tool in tools_list}