import __main__
import asyncio
import inspect
import aiohttp
from langchain.tools import tool
from typing import Dict, List, Literal
from langchain_core.tools import BaseTool
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from bs4 import BeautifulSoup
import json
import fitz  # PyMuPDF
import os
import ssl



@tool
def get_capital(country: str) -> str:
    """Get the capital of a country."""
    capitals = {"France": "Paris", "Germany": "Berlin"}
    return capitals.get(country, "Unknown")

@tool
def get_population(country: str) -> str:
    """Get the population of a country."""
    pops = {"France": 68000000, "Germany": 83000000}
    return str(pops.get(country, 0))

@tool
def calculate_difference(values: str) -> str:
    """Calculate the absolute difference between two numbers provided as a comma-separated string."""
    a, b = map(int, values.split(","))
    return str(abs(a - b))

@tool
def search_and_scrape_web(input: str, max_results: int = 10) -> list[dict]:
    """Search the web and scrape text content from the resulting URLs."""
    return asyncio.run(_search_and_scrape_web(input, max_results=max_results, type="text"))

@tool
def search_and_scrape_news(input: str, max_results: int = 10) -> list[dict]:
    """Search the news and scrape content from the resulting URLs."""   
    return asyncio.run(_search_and_scrape_web(input, max_results=max_results, type="news"))


ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def _search_and_scrape_web(input: str, max_results: int, type: Literal["text", "news"]) -> list[dict]:
    query = input.strip()
    results = []
    tasks = []

    async with aiohttp.ClientSession() as session:
        with DDGS() as ddgs:
            max_retries = 5
            for current_attempt in range(max_retries + 1):
                try:
                    search_results = ddgs.text(query, max_results=max_results) if type == "text" else ddgs.news(query, max_results=max_results)
                    for r in search_results:
                        url = r.get("href") or r.get("url")
                        if not url:
                            continue
                        tasks.append(download_and_parse_article(session, url))
                    completed = await asyncio.gather(*tasks, return_exceptions=True)
                    for res in completed:
                        if isinstance(res, Exception):
                            results.append({"url": "", "content": "", "error": str(res)})
                        elif res:
                            results.append(res)
                    break
                except DuckDuckGoSearchException as e:
                    if current_attempt < max_retries:
                        backoff_time = 2 ** current_attempt
                        print(f"{str(e)}. Retrying in {backoff_time} seconds...")
                        await asyncio.sleep(backoff_time)
                    else:
                        raise e

    return results

async def fetch(session: aiohttp.ClientSession, url: str) -> bytes | str:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30, sock_connect=5), ssl=ssl_context) as response:
            response.raise_for_status()
            return await response.content.read() if url.lower().endswith(".pdf") else await response.text()
    except Exception as e:
        raise e

async def download_and_parse_article(session: aiohttp.ClientSession, url: str) -> dict:
    try:
        response = await fetch(session, url)
        if isinstance(response, bytes):
            content = await asyncio.to_thread(parse_pdf, response)
        else:
            content = await asyncio.to_thread(parse_html, response)
        return {"url": url, "content": content}
    except Exception as e:
        return {"url": url, "content": "", "error": str(e)}

def parse_html(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return "\n".join(line for line in text.splitlines() if line.strip())

def parse_pdf(response: bytes) -> str:
    os.makedirs("tmp", exist_ok=True)
    temp_path = "tmp/temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(response)
    doc = fitz.open(temp_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

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
    print(search_and_scrape_web("Tesla last quarter performance"))
