from utils.logutil import setup_logger
setup_logger()


from langgraph.graph import StateGraph, END

from components.agents import planner_agent,researcher_executor,prepare_answer
from utils.common import AgentState
# ---------------- ROUTER ----------------


# class AgentState(TypedDict):
#     query: str
#     scratchpad: str
#     next_task: str
#     task_input: str
#     result: str
# high_level_tools = """
# [
#   {
#     "name": "internet_researcher",
#     "description": "Useful for answering questions by searching the internet",
#     "input": "A detailed question or topic to search"
#   },
#   {
#     "name": "news_researcher",
#     "description": "Useful for answering questions by searching news articles",
#     "input": "A detailed question or topic to search"
#   },
#   {
#     "name": "prepare_answer",
#     "description": "Useful for preparing the final answer from collected information",
#     "input": "No input required — the agent will use the full scratchpad to prepare the answer"
#   }
# ]
# """


# def route_dummy(state):
#     return None


def route(state):
    """Routing function: Determines which executor should handle the current task."""
    if state["next_task"] == "plan":
        return "plan"
    elif state["next_task"] == "internet_researcher" or state["next_task"] == "news_researcher":
        return "research"
    elif state["next_task"] == "prepare_answer":
        return "answer"
    else:
        raise ValueError(f"Unknown task: {state['next_task']}")
# ---------------- GRAPH ----------------

graph = StateGraph(AgentState)

graph.add_node("plan", planner_agent)
graph.add_node("research", researcher_executor)
graph.add_node("answer", prepare_answer)
# graph.add_node("math", math_executor)
# graph.add_node("toolnode", ToolNode(TOOLS))
# graph.add_node("router", route_dummy)
graph.set_entry_point("plan")
graph.add_conditional_edges("plan", route, {
    "research": "research",
    "answer": "answer",
    "plan": "plan"
})
graph.add_edge("research", "plan")
graph.add_edge("answer", END)

# def is_done(state):
#     """Check if all tasks are completed."""
#     return state["current_step"] >= len(state["tasks"])

# graph.add_conditional_edges("researcher", is_done, {True: END, False: "router"})
# graph.add_conditional_edges("cot", is_done, {True: END, False: "router"})
# graph.add_conditional_edges("math", is_done, {True: END, False: "router"})

app = graph.compile()
# print(app.get_graph().draw_ascii())


with open("graph.png", "wb") as f:
    print(f'writing file {f.name}')
    f.write(app.get_graph().draw_mermaid_png())

# ---------------- RUN ----------------
# query = "Find Tesla's last quarter performance, get capital of France, and calculate revenue change from 10B to 12B."
query = "Find Google's 2024 year performance, how is company doing and what are there immediate plan to get more market share."
# query = "What is apple inc up to? What are new features of the new products its going to launch?"
result = app.invoke({"query": query})


print(result["scratchpad"])

print("\n\n\n\n--------------------------Multi-Agent Routed Execution:")

print(result["result"])
# for r in result["results"]:
    # print(" -", r)
print("--------------------------")

