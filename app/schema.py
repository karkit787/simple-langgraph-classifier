from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class ClassifierState(BaseModel):
    """State for conversation workflow with classifier routes"""
    messages: Annotated[list[AnyMessage], add_messages]
    user_intent: str
    user_request: str
    confidence_level: float
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)