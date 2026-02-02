import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict

from.model import llm, confidence_threshold
from .prompts import CLASSIFIER_SYSTEM, CODING_SYSTEM, SUMMARIZATION_SYSTEM, PLANNING_SYSTEM, GENERAL_SYSTEM
from .schema import ClassifierState

def classifier_node(state: ClassifierState) -> Dict[str, any]:
    """Classifier node that analyzes user request to determine intent and confidence level"""
    retry_count = state.retry_count
    max_retries = state.max_retries
    user_request = state.messages[-1].content

    response = llm.invoke([SystemMessage(content=CLASSIFIER_SYSTEM), HumanMessage(content=user_request)])
    data = json.loads(response.content)
    intent = data["user_intent"]
    confidence = data["confidence_level"]

    if confidence < confidence_threshold:
        retry_count += 1

    return {
        "user_intent": intent,
        "user_request": user_request,
        "confidence_level": confidence,
        "retry_count": retry_count + 1
    }


def coding_node(state: ClassifierState) -> Dict[str, any]:
    """Handles coding questions"""
    response = llm.invoke([SystemMessage(content=CODING_SYSTEM), HumanMessage(content=state.user_request)])

    return {
        "messages": [response]
    }


def summarizaton_node(state: ClassifierState) -> Dict[str, any]:
    """Handles summarization requests"""
    response = llm.invoke([SystemMessage(content=SUMMARIZATION_SYSTEM), HumanMessage(content=state.user_request)])
    
    return {
        "messages": [response]
    }


def planning_node(state: ClassifierState) -> Dict[str, any]:
    """Handles planning guides"""
    response = llm.invoke([SystemMessage(content=PLANNING_SYSTEM), HumanMessage(content=state.user_request)])

    return{
        "messages": [response]
    }


def general_node(state: ClassifierState) -> Dict[str, any]:
    """Handles general questions for low confidence levels"""
    response = llm.invoke([SystemMessage(content=GENERAL_SYSTEM), HumanMessage(content=state.user_request)])
    
    return{
        "messages": [response]
    }