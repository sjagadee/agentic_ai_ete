from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,AIMessage, HumanMessage, AnyMessage
from langchain_core.tools import tool

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages ## used for adding messages to the list in state (reducer)

# use this to import tool_condition and ToolNode
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from typing import Annotated
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# set the project name for tracing
os.environ["LANGCHAIN_PROJECT"] = "ReAct Agent"

# Initialize open ai model using init_chat_model (newer way of setting up models in langchain)
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai",
    temperature=0.7,
)

# Adding tools

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

api_wrapper_wikipedia = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia)

tvaily_tool = TavilySearch()

# custom tool - for weather

@tool
def get_weather_tool(city: str):
    """
    Given a city name, this function will provide weather information about the city.
    Uses Weatherstack API to fetch the weather.
    """
    url = f'https://api.weatherstack.com/current?access_key={os.getenv("WEATHER_API_KEY")}&query={city}'

    response = requests.get(url)

    return response.json()

@tool
def add_tool(a: int, b: int) -> int:
    """Add two numbers given two inputs."""
    return a + b

@tool
def subtract_tool(a: int, b: int) -> int:
    """Subtract two numbers given two inputs."""
    return a - b

@tool
def multiply_tool(a: int, b: int) -> int:
    """Multiply two numbers given two inputs."""
    return a * b

@tool
def divide_tool(a: int, b: int) -> int:
    """Divide two numbers given two inputs."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# let us create a list of tools
tools = [arxiv_tool, wikipedia_tool, tvaily_tool, get_weather_tool, add_tool, subtract_tool, multiply_tool, divide_tool]

# bind the tools to the model
model_with_tools = model.bind_tools(tools=tools)

# let us build the State (chat state)

class ChatState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]

# let us implement the chat node
def chat_node(state: ChatState) -> ChatState:
    response = model_with_tools.invoke(state.messages)
    state.messages = state.messages + [response]
    return state


# let us implement the tool node
tools_node = ToolNode(tools)

def make_workflow_with_tools():

    # now lets build a graph with tools
    # initialize the graph
    graph = StateGraph(ChatState)

    # add nodes
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tools_node)

    # add edges
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    graph.add_edge("chat", END)

    # compile the graph
    agent_workflow_with_tools = graph.compile()

    return agent_workflow_with_tools

agent = make_workflow_with_tools()

##
"""
To run the agent and use LangGraph Studio - run the following command
"langgraph dev"

Note: Memory is not yet supported in LangGraph Studio 

"""