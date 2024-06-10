
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[HumanMessage | AIMessage]]
