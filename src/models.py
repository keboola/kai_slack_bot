from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    query: str
    