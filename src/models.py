from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]

