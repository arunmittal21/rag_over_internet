# Multi-Agent ReAct System with Planner and Routed Executors using ToolNode

import json
import re
# from langgraph.visualization import visualize

from utils.common import llm, rerank_segments, chunk_documents
from components.tools import TOOL_MAP ,get_tool_manifest_json, search_and_scrape_web, search_and_scrape_news


high_level_tools = """
[
  {
    "name": "internet_researcher",
    "description": "Given input will be searched on the internet, Content from first few links will be summarized and returned",
    "input": "Google search friendly query. Try to be specific and focus on one topic at a time"
  },
  {
    "name": "news_researcher",
    "description": "Given input will be searched in news articles, Content from first few links will be summarized and returned",
    "input": "Google search friendly query. Try to be specific and focus on one topic at a time"
  },
  {
    "name": "prepare_answer",
    "description": "Useful for preparing the final answer from collected information",
    "input": "No input required — the agent will use the full scratchpad to prepare the answer"
  }
]
"""

def planner_agent(state):
    """Planner agent: Breaks down a high-level query into subtasks routed to specific tools."""
    query = state["query"]
    scratchpad = state.get("scratchpad", "")

    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a planner agent equipped with high-level tools to solve user questions.
Based on the query and scratchpad, decide the next task and pick one tool to execute it. Try to be specific in your input to the tool to get the best results.
Do not keep repeating the same tool with same inputs if it has already been used in the scratchpad, as it will not give you different result.

You can use these tools:
{high_level_tools}

Use this exact format for your reasoning:
Task: Describe your thinking or what you need to do.
Action: The tool name to use (exact match from the above list).
Action Input: The input string for that tool. Be as specific as possible as this will be the only input passed to the tool.
Action Output: (Leave this blank — it will be filled after the tool is called.)

Guidelines:
- Think step-by-step.
- Use one tool per step.
- Call `prepare_answer` only when enough information has been gathered.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}

Current scratchpad:
{scratchpad}
"""

    response = llm.invoke(prompt).content.strip()
    scratchpad += "\n" + response + "\n"
    print("✅ Planner reasoning:\n", response)
    # Match Action and Action Input
    match = re.search(r"Action:\s*(\w+)\s*Action Input:\s*\"?(.+?)\"?(?:\s|$)", response, re.DOTALL)
    if match:
        executor_name = match.group(1).strip()
        executor_input = match.group(2).strip()
        # print(f"✅ Planner selected: {executor_name} with input: {executor_input}")
        return {
            **state,
            "next_task": executor_name,
            "task_input": executor_input,
            "scratchpad": scratchpad,
            "current_step": state.get("current_step", 0) + 1
        }
    else:
        raise ValueError("❌ No valid 'Action' and 'Action Input' found in planner output.")


def researcher_executor(state):
    """Executor for 'internet_researcher' or 'news_researcher' tools."""
    task_input: str = state.get("task_input", "")
    task = state.get("next_task", "")
    tool = None
    if task == 'internet_researcher':
        tool = search_and_scrape_web
    elif task == 'news_researcher':
        tool = search_and_scrape_news
    else:
        raise ValueError(f"Unknown researcher tool requested: {task}")

    if tool is None:
        raise ValueError(f"Tool not found for task: {task}")
    
    tool_args = {"input": task_input, "max_results": 20}
    tool_output = tool.invoke(tool_args)
    chunked_docs = chunk_documents([doc['content'] for doc in tool_output], chunk_size=1000, overlap=50)
    reranked_docs = rerank_segments(chunked_docs, query=task_input)
    tool_output_reranked = "\n".join(reranked_docs)
    # print(f"🧪 {task} tool output:\n", tool_output)
    research_prompt = f"""
You are a summarizer. Here is the raw content from the search results:
{tool_output_reranked[:7800]}

Provide a clear and concise summary of your findings in less than 1000 tokens. The topic is:- {task_input}.
"""
    summary = llm.invoke(research_prompt).content
    print(f"🧪 {task} tool summary:\n", summary)
    return {
        **state,
        "next_task": "plan",  # Go back to planner after research
        "task_input": "",
        "scratchpad": state["scratchpad"] + f"{summary}\n",
        "current_step": state["current_step"] + 1,
    }


def prepare_answer(state):
    """Final step: Prepares answer based on full scratchpad and query."""
    query = state["query"]
    scratchpad = state["scratchpad"]

    final_prompt = f"""
You are a helpful assistant. Use the research notes below to provide a complete and final answer to the original query.
Do not make up any information. If the information is insufficient, state that clearly.

Query: {query}

Scratchpad:
{scratchpad}

Final Answer:
"""
    final_response = llm.invoke(final_prompt).content
    return {
        **state,
        "result": final_response,
        "scratchpad": scratchpad + f"\nFinal Answer: {final_response}",
        "current_step": state["current_step"] + 1
    }


