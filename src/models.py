from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


class PineconeComparator(str, Enum):
    """Enumerator of the comparison operators."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    LIKE = "like"
    IN = "in"
    NIN = "nin"
