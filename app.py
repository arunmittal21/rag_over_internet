# Multi-Agent ReAct System with Planner and Routed Executors using ToolNode

from langgraph.graph import StateGraph, END
#from langgraph.visualization import visualize

from components.agents import planner_agent,researcher_executor,cot_executor,math_executor
from components.common import AgentState
# ---------------- ROUTER ----------------


def route_dummy(state):
    return None


def route(state):
    """Routing function: Determines which executor should handle the current task."""
    executor_type = state["tasks"][state["current_step"]]["executor"]
    return executor_type

# ---------------- GRAPH ----------------

graph = StateGraph(AgentState)

graph.add_node("plan", planner_agent)
graph.add_node("researcher", researcher_executor)
graph.add_node("cot", cot_executor)
graph.add_node("math", math_executor)
# graph.add_node("toolnode", ToolNode(TOOLS))
graph.add_node("router", route_dummy)

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

app = graph.compile()
# visualize(graph).render("agent_workflow", format="png", view=True)

# ---------------- RUN ----------------
query = "Find Tesla's last quarter performance, get capital of France, and calculate revenue change from 10B to 12B."
result = app.invoke({"query": query})
print("\nMulti-Agent Routed Execution:")
for r in result["results"]:
    print(" -", r)

