# Multi-Agent ReAct System with Planner and Routed Executors using ToolNode

from langchain.tools import tool
# from langgraph.visualization import visualize


@tool
def get_capital(country: str) -> str:
    """Return the capital of a given country. Supported: France, Germany."""
    capitals = {"France": "Paris", "Germany": "Berlin"}
    return capitals.get(country, "Unknown")

@tool
def get_population(country: str) -> str:
    """Return the population of a given country as a string. Supported: France, Germany."""
    pops = {"France": 68000000, "Germany": 83000000}
    return str(pops.get(country, 0))

@tool
def calculate_difference(values: str) -> str:
    """Calculate absolute difference between two integers passed as comma-separated values (e.g. '10,20')."""
    a, b = map(int, values.split(","))
    return str(abs(a - b))


from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from newspaper import Article

@tool
def search_and_scrape_web(input: str) -> str:
    """
    Search the web for a given query and return the top 3 URLs with page content summary.
    Input should be the search query.
    """
    query = input.strip()
    results = []
    
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=3):
            url = r.get("href") or r.get("url")
            if not url:
                continue
            try:
                article = Article(url)
                article.download()
                article.parse()
                text = article.text.strip()
                results.append({"url": url, "content": text[:1500] if text else "No content found."})
            except Exception as e:
                results.append({"url": url, "content": f"Error: {str(e)}"})
    return str(results)


TOOLS = [get_capital, get_population, calculate_difference]
TOOL_MAP = {t.name: t for t in TOOLS}

