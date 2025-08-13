from typing import Dict, List, Literal

from pydantic import BaseModel


class RAGState(BaseModel):
    query: str
    subquestions: List[str] = []
    answers: Dict[str, str] = {}
    current_depth: int = 1
    route_decision: str = ""
    user_decision: str = ""
    retry_times: int = 0
    human_suggestion: str = ""
