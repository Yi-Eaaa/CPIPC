from typing import TypedDict, List, Dict, Optional

class QuestionNode(TypedDict):
    id: str
    question: str
    answer: Optional[str]
    knowledge_gap: Optional[str]
    depth: int
    provide_info: Optional[str]
    children: List["QuestionNode"]

class RAGState(TypedDict):
    query: str
    question_queue: List[str]
    answers: Dict[str, str]
    current_depth: int
    route_decision: str
    tree: QuestionNode
    node_map: Dict[str, QuestionNode]
    root_query: str
    human_suggestion: Optional[str]
