from pydantic import BaseModel
from langchain_core.output_parsers import BaseOutputParser
from typing import List, Optional, Sequence

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return lines

