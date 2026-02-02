from typing import Literal
from langgraph.graph import StateGraph, START, END
from .schema import ClassifierState
from .nodes import classifier_node, coding_node, summarizaton_node, planning_node, general_node

def route_by_intent(state: ClassifierState) -> Literal["coding", "summarization", "planning", "classifier", "general"]:
    """
    Routing function for conditional edges.

    This function determines which node to execute next based on the current state.
    It must return a string that matches one of the node names or END.

    Args:
        state: Current conversation state

    Returns:
        The name of the next node to execute
    """
    intent = state.user_intent if state.user_intent is not None else "unclassified"

    if intent == "coding":
        return "coding"
    elif intent == "summarization":
        return "summarization"
    elif intent == "planning":
        return "planning"
    else:
        if state.retry_count < state.max_retries:
            return "classifier"
        return "general"
    

def create_graph() -> StateGraph:
    """Creates a graph with conditional routing based on user intent."""
    graph = StateGraph(ClassifierState)

    graph.add_node("classifier_node", classifier_node)
    graph.add_node("coding_node", coding_node)
    graph.add_node("summarization_node", summarizaton_node)
    graph.add_node("planning_node", planning_node)
    graph.add_node("general_node", general_node)

    graph.add_edge(START, "classifier_node")
    graph.add_conditional_edges("classifier_node",
                                route_by_intent, {
                                    "coding": "coding_node",
                                    "summarization": "summarization_node",
                                    "planning": "planning_node",
                                    "classifier": "classifier_node",
                                    "general": "general_node"
                                })
    graph.add_edge("coding_node", END)
    graph.add_edge("summarization_node", END)
    graph.add_edge("planning_node", END)
    graph.add_edge("general_node", END)

    return graph.compile()