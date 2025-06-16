# Multi-Agent ReAct System with Planner and Routed Executors using ToolNode

import json
import re
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, List, Union
from langchain.callbacks.base import BaseCallbackHandler

MODEL_ID= 'meta.llama3-70b-instruct-v1:0'
class SimpleLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("📥 Prompt sent:", prompts)

    def on_llm_end(self, response, **kwargs):
        print("📤 Response received:", response)


# ---------------- TOOLS ----------------
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

tools = [get_capital, get_population, calculate_difference]

# ---------------- PLANNER ----------------

planner_llm = ChatBedrock(
    model_id=MODEL_ID,
    region_name="us-east-1",
    temperature=0,
    callbacks=[SimpleLogger()],
)

def planner_agent(state):
    """Planner agent: Breaks down a high-level query into structured subtasks with designated executor types."""
    query = state["query"]
    prompt = f"""
You are a planner. Break down the query into high-level subtasks with executor type.
Query: {query}
Respond in JSON list like:
[
  {{"executor": "researcher", "task": "Find Tesla's recent performance."}},
  {{"executor": "cot", "task": "Get capital of France."}},
  {{"executor": "math", "task": "Calculate revenue change from Q1 to Q2."}}
]
"""
    response = planner_llm.invoke(prompt)
    subtasks = json.loads(response.content)
    return {"tasks": subtasks, "current_step": 0, "results": [], "query": query}

# ---------------- EXECUTORS ----------------
llm = ChatBedrock(
    model_id=MODEL_ID,
    region_name="us-east-1",
    temperature=0,
    callbacks=[SimpleLogger()],
)

REACT_PROMPT = """
You are a chain-of-thought executor with tools.
Task: {task}
Available tools: get_capital, get_population, calculate_difference
Use this format:
Thought: ...
Action: ...
Action Input: ...
Observation: ...
Final Answer: ...
"""

def cot_executor(state):
    """Chain-of-thought (CoT) executor: Uses tools and reasoning steps to solve tasks step-by-step."""
    task = state["tasks"][state["current_step"]]["task"]
    scratchpad = ""
    for _ in range(5):
        prompt = REACT_PROMPT.format(task=task) + scratchpad
        output = llm.invoke(prompt).content.strip()
        if "Final Answer:" in output:
            answer = output.split("Final Answer:")[1].strip()
            return {**state, "current_step": state["current_step"] + 1, "results": state["results"] + [answer]}
        m = re.search(r"Action: (\w+)\s*Action Input: (.+)", output)
        if m:
            tool_name, tool_input = m.group(1).strip(), m.group(2).strip()
            tool_fn = next(t for t in tools if t.name == tool_name)
            result = tool_fn.invoke(tool_input)
            scratchpad += f"{output}\nObservation: {result}\n"
    return state

RESEARCH_PROMPT = """
You are a researcher. You should look up or synthesize knowledge from known facts.
Task: {task}
Respond with a summary of your findings.
"""

def researcher_executor(state):
    """Researcher agent: Answers questions based on synthesis or known information."""
    task = state["tasks"][state["current_step"]]["task"]
    response = llm.invoke(RESEARCH_PROMPT.format(task=task)).content.strip()
    return {**state, "current_step": state["current_step"] + 1, "results": state["results"] + [response]}

MATH_PROMPT = """
You are a math-focused executor. Solve the task directly.
Task: {task}
"""

def math_executor(state):
    """Math agent: Directly solves numerical or arithmetic tasks."""
    task = state["tasks"][state["current_step"]]["task"]
    response = llm.invoke(MATH_PROMPT.format(task=task)).content.strip()
    return {**state, "current_step": state["current_step"] + 1, "results": state["results"] + [response]}

# ---------------- ROUTER ----------------
def route(state):
    """Routing function: Determines which executor should handle the current task."""
    executor_type = state["tasks"][state["current_step"]]["executor"]
    return executor_type

# ---------------- GRAPH ----------------
class AgentState(TypedDict):
    query: str
    tasks: List[dict]
    current_step: int
    results: List[Union[str, dict]]

graph = StateGraph(AgentState)

graph.add_node("plan", planner_agent)
graph.add_node("researcher", researcher_executor)
graph.add_node("cot", cot_executor)
graph.add_node("math", math_executor)
graph.add_node("toolnode", ToolNode(tools))
graph.add_node("router", route)

graph.set_entry_point("plan")
graph.add_edge("plan", "router")

graph.add_conditional_edges("router", route, {
    "researcher": "researcher",
    "cot": "cot",
    "math": "math"
})

def is_done(state):
    """Check if all tasks are completed."""
    return state["current_step"] >= len(state["tasks"])

graph.add_conditional_edges("researcher", is_done, {True: END, False: "router"})
graph.add_conditional_edges("cot", is_done, {True: END, False: "router"})
graph.add_conditional_edges("math", is_done, {True: END, False: "router"})

# graph.set_finish_point(END)
app = graph.compile()
# from langgraph.visualization import visualize
# visualize(graph).render("agent_workflow", format="png", view=True)

# ---------------- RUN ----------------
query = "Find Tesla's last quarter performance, get capital of France, and calculate revenue change from 10B to 12B."
result = app.invoke({"query": query})
print("\nMulti-Agent Routed Execution:")
for r in result["results"]:
    print(" -", r)
