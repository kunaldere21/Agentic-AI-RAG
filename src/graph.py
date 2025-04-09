# graph.py
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict

class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]

def build_graph(agent_node, retriever_tool, generate_node, rewrite_node, web_search_node, grade_node):
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("retrieve", ToolNode([retriever_tool]))
    graph.add_node("generate", generate_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("web_search", web_search_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {
        "tools": "retrieve",
        END: END,
    })

    graph.add_conditional_edges("retrieve", grade_node, {
        "generate": "generate",
        "rewrite": "rewrite",
        "web_search": "web_search"
    })

    graph.add_edge("generate", END)
    graph.add_edge("web_search", 'generate')
    graph.add_edge("rewrite", "agent")

    return graph.compile()
