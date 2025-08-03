import os
import sys
import uuid
from typing import List, Dict, Literal

from pydantic import BaseModel
from langgraph.graph import END, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from src.llm.api import SiliconFlowAPI
from src.config.config import GLOABLE_CONFIG
from src.llm.operate import hybrid_response
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.bm25_retriever import BM25Retriever
from src.rag.logger import Logger
from src.rag.base import RAGState

class MiniRAG:
    def __init__(self):
        # Set up paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if project_root not in sys.path:
            sys.path.append(project_root)

        # Initialize core components
        self.logger = Logger("MiniRAG").get_logger()
        self.llm = SiliconFlowAPI(GLOABLE_CONFIG["chat_api_key"])
        self.model = GLOABLE_CONFIG["chat_model"]
        self.checkpointer = MemorySaver()
        self.retriever_vector = DenseRetriever()
        self.retriever_bm25 = BM25Retriever()
        self.workflow = self._build_graph()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("check", self.check_node)
        graph.add_node("planner", self.planner_node)
        graph.add_node("combine", self.combine_node)
        graph.add_node("exit", self.exit_node)

        graph.set_entry_point("check")
        graph.add_conditional_edges("check", self.check_route, {
            "planner": "planner",
            "combine": "combine"
        })
        graph.add_edge("planner", "combine")
        graph.add_edge("exit", END)
        return graph

    def check_node(self, state: RAGState) -> RAGState:
        self.logger.info(f"🔍 Checking query: {state.query} (depth {state.current_depth})")
        vector_docs = self.retriever_vector.retrieve(state.query, 1)
        bm25_docs = self.retriever_bm25.retrieve(state.query, 1)
        answer = hybrid_response(state.query, vector_docs, bm25_docs)

        self.logger.info(f"Answer: {answer}")

        if answer.strip().upper().startswith("INSUFFICIENT"):
            self.logger.warning(f"⚠️ Cannot answer directly: {state.query}")
            state.answers[state.query] = ""
            state.route_decision = "combine" if state.current_depth >= 3 else "planner"
        else:
            state.answers[state.query] = answer
            state.route_decision = "combine"

        return state

    def check_route(self, state: RAGState) -> str:
        self.logger.info(f"🚦 Routing decision: {state.query} -> {state.route_decision}")
        return state.route_decision

    def planner_node(self, state: RAGState) -> RAGState:
        self.logger.info(f"🧩 Planning subquestions for: {state.query} (depth {state.current_depth})")

        prompt = (
            f"User suggestion: {state.human_suggestion}\n\n" if state.human_suggestion else ""
        ) + f"Please decompose the following complex question into simpler, answerable subquestions:\n{state.query}"

        subqs = [q.strip() for q in self.llm.chat(self.model, prompt).split("\n") if q.strip()]
        state.subquestions = subqs
        self.logger.info(f"📌 Subquestions generated: {subqs}")

        for subq in subqs:
            if subq not in state.answers:
                self.logger.info(f"🔁 Recursively answering: {subq}")
                sub_state = RAGState(
                    query=subq,
                    answers=state.answers,
                    current_depth=state.current_depth + 1
                )
                config = {"configurable": {"thread_id": uuid.uuid4()}}
                result = self.app.invoke(sub_state, config=config)
                state.answers.update(RAGState(**result).answers)

        combined_answer = self._combine_answers(state)
        if combined_answer:
            state.answers[state.query] = combined_answer

        return state

    def _combine_answers(self, state: RAGState) -> str:
        all_sub_answers = "\n".join(f"{k}: {v}" for k, v in state.answers.items() if k != state.query)
        if not all_sub_answers:
            self.logger.warning("❗ No subanswers found to combine.")
            return ""
        prompt = f"Given the answers below, synthesize a coherent response to the query: {state.query}\n\n{all_sub_answers}"
        combined = self.llm.chat(self.model, prompt)
        self.logger.info(f"🧠 Final combined answer: {combined}")
        return combined

    def combine_node(self, state: RAGState) -> Command[Literal["check", "exit"]]:
        if state.current_depth > 1:
            self.logger.info("➡️ Not top-level, skipping human-in-the-loop.")
            return state

        self.logger.info("✅ Top-level reached, requesting user decision.")
        while True:
            decision = input("👉 Enter 'approve' or 'retry': ").strip().lower()
            if decision in {"approve", "retry"}:
                break
            print("❗ Invalid input, try again.")

        if decision == "approve":
            return Command(goto="exit", update={"decision": "approve"})

        suggestion = input("💡 Enter suggestion for rephrasing (optional): ").strip()
        return Command(goto="check", update=self._reset_for_retry(state, suggestion))

    def exit_node(self, state: RAGState) -> RAGState:
        return state

    def _reset_for_retry(self, state: RAGState, suggestion: str) -> dict:
        return {
            "query": state.query,
            "subquestions": [],
            "answers": {},
            "current_depth": 1,
            "route_decision": "",
            "user_decision": "",
            "retry_times": state.retry_times + 1,
            "human_suggestion": suggestion
        }

    def run(self, query: str):
        self.logger.info(f"🚀 Starting workflow for query: {query}")
        init_state = RAGState(query=query)
        config = {"configurable": {"thread_id": uuid.uuid4()}}
        result = self.app.invoke(init_state, config=config)

        if isinstance(result, dict) and "__interrupt__" in result:
            interrupt_data = result["__interrupt__"]
            print("🛑 Interrupted:", interrupt_data)
            decision = input("👉 Enter 'approve' or 'reject': ").strip().lower()
            result = self.app.invoke(Command(resume=decision), config=config)

        final_state = RAGState(**result)
        print("\n✅ Final Answer:\n", final_state.answers.get(query))
        print("\n📚 Subquestions and Answers:")
        for k, v in final_state.answers.items():
            if k != query:
                print(f"- {k}: {v}")


if __name__ == "__main__":
    query = "Who are the protagonists in *A Christmas Carol* and *What I Worked On* respectively?"
    rag_app = MiniRAG()
    rag_app.run(query)
