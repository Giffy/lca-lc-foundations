# Requires a valid LANGSMITH_API_KEY
# run it using:  uv run langgraph dev
# and open gui in browser: http://127.0.0.1:2024/


from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
#from langgraph.checkpoint.memory import InMemorySaver

model_name="granite4:1b"
model_url=os.getenv('OLLAMA_HOST')

model = ChatOllama(
    model=model_name,
    api_base=model_url
)

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt,
    #checkpointer=InMemorySaver()
)