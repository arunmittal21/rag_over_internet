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
# from langgraph.visualization import visualize

from components.common import llm
from components.tools import TOOL_MAP

def planner_agent(state):
    """Planner agent: Breaks down a high-level query into structured subtasks with designated executor types."""
    query = state["query"]
    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Respond ONLY with a JSON array of subtasks. No explanation or formatting
<|eot_id|><|start_header_id|>user<|end_header_id|>    
You are a planner. Break down the query into high-level subtasks with executor type.
Query: {query}
Respond ONLY with valid JSON:
[
  {{"executor": "researcher", "task": "Find Tesla's recent performance."}},
  {{"executor": "cot", "task": "Get capital of France."}},
  {{"executor": "math", "task": "Calculate revenue change from Q1 to Q2."}}
]
"""
    response = llm.invoke(prompt).content
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if match:
        subtasks = json.loads(match.group(0))
    else:
        raise ValueError("❌ No valid JSON list found in planner response")
    return {"tasks": subtasks, "current_step": 0, "results": [], "query": query}

REACT_PROMPT = """
You are a chain-of-thought reasoning agent equipped with external tools to help solve user questions.

Your task is: {task}

You can use the following tools:

1. get_capital(country: str) -> str  
   - Description: Returns the capital of a given country.  
   - Supported countries: France, Germany  
   - Example Input: "France"

2. get_population(country: str) -> str  
   - Description: Returns the population of a given country as a string.  
   - Supported countries: France, Germany  
   - Example Input: "Germany"

3. calculate_difference(values: str) -> str  
   - Description: Calculates the absolute difference between two integers.  
   - Input should be a comma-separated string of two numbers.  
   - Example Input: "68000000,83000000"

Use the following format for your reasoning:

Thought: Describe your thinking or what you need to find out.
Action: The tool you want to use (must be exactly one of the tools above).
Action Input: The input to the tool (must follow the input format shown above).
Observation: (Leave this blank. It will be filled in after the tool is executed.)
... (Continue as needed)
Final Answer: Provide the final answer after all reasoning is complete.

Instructions:
- Think step-by-step and break down the task logically.
- **Do NOT fabricate the Observation — End your response when you need observation.**
- Use one tool per step.
- Only use supported countries and input formats.
- Stop when you have enough information to confidently give the Final Answer.
"""


def cot_executor(state):
    """Chain-of-thought (CoT) executor: Uses tools and reasoning steps to solve tasks step-by-step."""
    task = state["tasks"][state["current_step"]]["task"]
    scratchpad = ""
    for i in range(5):
        prompt = REACT_PROMPT.format(task=task) + scratchpad
        output = llm.invoke(prompt).content.strip()
        print(f'output from cot is {i}: {output}, scratchpad is {scratchpad}')
        if "Final Answer:" in output:
            answer = output.split("Final Answer:")[1].strip()
            return {**state, "current_step": state["current_step"] + 1, "results": state["results"] + [answer]}
        m = re.search(r'Action: (\w+)\s*Action Input: \"(.+)\"', output)
        if m:
            tool_name, tool_input = m.group(1).strip(), m.group(2).strip()
            print(f'calling tool {tool_name} with input {tool_input}')
            tool = TOOL_MAP.get(tool_name)
            if tool:
                result = tool.invoke(tool_input)
                scratchpad += f"{output}\nObservation: {result}\n"
            else:
                scratchpad += f"{output}\nObservation: Tool '{tool_name}' not found\n"
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


