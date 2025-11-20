import json
import random
from datetime import datetime, timedelta, timezone
from pprint import pprint
from typing import Literal
from langchain.tools import BaseTool, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama


Location = Literal[
    "Abuja", "Accra", "Cairo", "Dakar", "Gaborone", "Kigali", "Nairobi","Praia", "Pretoria"
]
time_diffs: dict[Location, float] = {
    "Abuja": 1,
    "Accra": 0,
    "Cairo": 2,
    "Dakar": 0,
    "Gaborone": 2,
    "Kigali": 2,
    "Nairobi": 3,
    "Praia": -1,
    "Pretoria": 2,
}

system_message_prompt_template = """
You're a helpful assistant.
You have access to the following tools:
{tools}

- Whenever the user asks for weather or time related information, you retrieve that information from the relevant tool 
and provide the user with the retrieved information.
- If the user requests for the time in a given location, you should first of all retrieve the location's 
  offset from UTC, and subsequently use the retrieved offset in retrieving the local time in that location.
"""
TemperatureUnit = Literal["Celsius", "Fahrenheit"]
model_name = "llama3.2:3b"
temperature = 0.0

def generate_temperature(min_temp: float = 8, max_temp: float = 45) -> float:
    """
    Generate random temperature value between min_temp and max_temp.
    :param min_temp: Minimum temperature value.
    :param max_temp: Maximum temperature value.
    :return: float value between min_temp and max_temp, inclusive, rounded to 1 decimal place.
    """
    return round(random.uniform(min_temp, max_temp), 1)

def get_tools_info(tools_list: list[BaseTool]) -> dict[str, str]:
    """
    Retrieve tools information from a list of tools.
    :param tools_list: The list of tools whose information is to be retrieved.
    :return: A dictionary with tools information, containing each tool's name, description, and schema
    """
    tools_info = {}

    for t in tools_list:
        tools_info[t.name] = t.description

    return tools_info

def generate_prompt_to_runnable_chain(template: ChatPromptTemplate, runnable: Runnable) -> Runnable:
    """
    Generates a prompt_template -> Runnable chain
    :param template: The prompt_template to use.
    :param runnable: The runnable to use; e.g. llm.
    :return: A Runnable (prompt_template -> Runnable chain).
    """
    return template | runnable

@tool()
def get_location_temperature(location: Location, units: TemperatureUnit = "Celsius") -> str:
    """
    Retrieve the temperature in a given African capital city
    :param location: Location to get the temperature for
    :param units: Temperature unit - must be one of "Celsius", and "Fahrenheit
    :return: A string containing the temperature, units, and location
    """
    if location not in time_diffs:
        raise ValueError(f"Invalid location {location}")

    ambient_temperature = generate_temperature()
    return f"The temperature in {location} is: {ambient_temperature} {units}"

@tool()
def get_local_datetime(offset_from_utc: float) -> datetime:
    """
    Retrieve the current local datetime in a given timezone
    :param offset_from_utc: Time offset from UTC
    :return: Current datetime in the specified timezone
    """
    return datetime.now(tz=timezone(timedelta(hours=offset_from_utc)))

@tool()
def get_offset_from_utc(location: Location) -> float:
    """
    Get the time offset from UTC for the specified location
    :param location: The location to get its offset from UTC.
       must be one of "Abuja", "Accra", "Cairo", "Dakar", "Gaborone", "Kigali", "Nairobi","Praia", and "Pretoria"
    :return: float representing the time offset from UTC
    """
    if location not in time_diffs:
        raise ValueError(f"Invalid location {location}")

    return time_diffs[location]

available_tools = [get_location_temperature, get_local_datetime, get_offset_from_utc]
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_message_prompt_template),
        ("human", "{input_prompt}")
    ]
)
llm = ChatOllama(model=model_name, temperature=temperature)
llm_with_tools = llm.bind_tools(tools=available_tools)

locale: Location = "Praia"


# Weather queries
weather_without_tool = generate_prompt_to_runnable_chain(
    template=prompt_template,
    runnable=llm
).invoke(
    {
        "input_prompt":f"What's the current weather in {locale}?",
        "tools": json.dumps(get_tools_info(available_tools), indent=4),
    }
)
pprint(f"Response to weather query without tool:\n{weather_without_tool}")

weather_with_tool = generate_prompt_to_runnable_chain(
    template=prompt_template,
    runnable=llm_with_tools
).invoke(
    {
        "input_prompt":f"What's the current weather in {locale}?",
        "tools": json.dumps(get_tools_info(available_tools), indent=4),
    }
)
pprint(f"Response to weather query with tool:\n{weather_with_tool}")
