# Multi-Agent ReAct System with Planner and Routed Executors using ToolNode

import __main__
import asyncio
import inspect
from langchain.tools import tool
from typing import Dict, List, Literal
# from langgraph.visualization import visualize
from langchain_core.tools import BaseTool
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException, DuckDuckGoSearchException
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import json
import fitz  # PyMuPDF
import time

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



@tool
def search_and_scrape_web(input: str,max_results: int=10) -> str:
    """Search the web for a given query and return the top 3 URLs with page content summary."""
    return asyncio.run(_search_and_scrape_web(input, max_results=max_results, type="text"))

@tool
def search_and_scrape_news(input: str, max_results: int=10) -> str:
    """Search the web for a given query and return the top 3 URLs with page content summary."""
    return asyncio.run(_search_and_scrape_web(input, max_results=max_results, type="news"))

async def _search_and_scrape_web(input: str, max_results: int, type: Literal["text", "news"]) -> str:
    """
    Search the web for a given query and return the top 3 URLs with page content summary.
    Input should be the search query.
    """
    query = input.strip()
    results = []
    task=[]
    with DDGS() as ddgs:
        max_retries = 5
        for current_attempt in range(max_retries + 1):
            try:
                for r in ddgs.text(query, max_results=max_results) if type == "text" else ddgs.news(query, max_results=max_results):
                    url = r.get("href") or r.get("url")
                    if not url:
                        continue
                    try:
                        task.append(asyncio.create_task(download_and_parse_article(url)))
                        # article = Article(url)
                        # article.download()
                        # article.parse()
                        # text = article.text.strip()
                        # results.append({"url": url, "content": text[:1500] if text else "No content found."})
                    except Exception as e:
                        results.append({"url": url, "content": f"Error: {str(e)}"})
                await asyncio.gather(*task)
                for res in task:
                    if res.result():
                        results.append(res.result())
                break
            except DuckDuckGoSearchException as e:
                if current_attempt < max_retries:
                    current_attempt += 1
                    # Exponential backoff
                    backoff_time = 2 ** current_attempt
                    print(f"{str(e)}. Retrying in {backoff_time} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    raise e

    return json.dumps(results, indent=2)


# async def download_and_parse_article(url: str) -> dict | None:
#     def sync_scrape():
#         article = Article(url)
#         article.download()
#         article.parse()
#         text = article.text.strip()
#         return {"url": url, "content": text} if text else None

#     try:
#         return await asyncio.to_thread(sync_scrape)
#     except Exception as e:
#         return {"url": url, "content": f"Error: {str(e)}"}

async def download_and_parse_article(url: str) -> dict | None:
    def sync_scrape():
        try:
            if url.lower().endswith(".pdf"):
                return parse_pdf(url)
            else:
                return parse_html(url)
        except Exception as e:
            return {"url": url, "content": f"Error: {str(e)}"}

    return await asyncio.to_thread(sync_scrape)


def parse_html(url: str) -> dict:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove unwanted tags
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Get visible text
    text = soup.get_text(separator="\n", strip=True)
    text = "\n".join(line for line in text.splitlines() if line.strip())

    return {"url": url, "content": text[:10000]} if text else {"url": url, "content": "No content found."}


def parse_pdf(url: str) -> dict:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    with open("tmp/temp.pdf", "wb") as f:
        f.write(response.content)

    doc = fitz.open("tmp/temp.pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    return {"url": url, "content": text[:10000]} if text else {"url": url, "content": "No content found in PDF."}



def find_tools_in_current_script():
    tools = []
    for name, obj in inspect.getmembers(__main__):
        if isinstance(obj, BaseTool):
            tools.append(obj)
    return tools

TOOLS = find_tools_in_current_script()
TOOL_MAP = {t.name: t for t in TOOLS}

def get_tool_manifest_json(tools: List[BaseTool] = TOOLS) -> List[Dict]:
    tool_info = []

    for tool in tools:
        args = []
        args_schema = getattr(tool, "args_schema", None)

        if args_schema and hasattr(args_schema, "model_fields"): 
            for field_name, field in args_schema.model_fields.items():
                field_info = {
                    "name": field_name,
                    "type": field.annotation.__name__ if isinstance(field.annotation, type) else str(field.annotation),
                }
                if field.description:
                    field_info["description"] = field.description
                args.append(field_info)

        tool_info.append({
            "name": tool.name,
            "description": getattr(tool, "description", "No description"),
            "args": args
        })

    return tool_info

if __name__ == "__main__":
    import json
    # print(json.dumps(get_tool_manifest_json(TOOLS), indent=2))
    print(search_and_scrape_web("Tesla last quarter performance"))



