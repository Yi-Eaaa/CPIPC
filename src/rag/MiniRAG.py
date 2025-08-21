
from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from src.llm.agent import Agent
from src.llm.prompts import PROMPTS
from src.config.config import GLOABLE_CONFIG
from src.llm.operate import hybrid_response, extract_entity, extract_triple, query_with_et, bm25_retrieve, dense_retrieve
from logger import Logger
from base import RAGState, QuestionNode
import uuid
import os
import json

class MiniRAG:

    def __init__(
        self,
        log_name: str = "MiniRAG",
        docs_set: str = "crag_data0",
        topk: int = 10,
        max_depth: int = 3,
        recursion_limit: int = 50,
        thread_id: Optional[str] = None,
    ):
        # init
        self.agent = Agent(GLOABLE_CONFIG["chat_api_key"])
        self.model = GLOABLE_CONFIG["chat_model"]
        self.docs_set = docs_set
        self.topk = topk
        self.max_depth = max_depth

        # config
        self.config = {
            "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
            "recursion_limit": recursion_limit,
        }
        # logger
        self.logger = Logger(log_name=log_name).get_logger()
        self.logger.info(
            "MiniRAG initialized. model=%s, docs_set=%s",
            self.model,
            docs_set,
        )

        # LangGraph app
        self.app = self._build_graph()
        self.save_graph_png()
        self.logger.info("LangGraph compiled")

    # ---------- State & Utils ----------
    def create_initial_state(self, query: str) -> RAGState:
        root_node: QuestionNode = {
            "id": str(uuid.uuid4()),
            "question": query,
            "depth": 1,
            "answer": None,
            "children": [],
        }
        return RAGState(
            query=query,
            question_queue=[query],
            answers={},
            current_depth=1,
            route_decision="",
            tree=root_node,
            node_map={query: root_node},
            root_query=query,
            human_suggestion=None,
            planning_nodes=[],
            planning_marks=[],
        )

    def print_tree(self, node: QuestionNode, indent: int = 0):
        if indent == 0:
            self.logger.info("🧭 Final Question Tree:")

        spacer = "  " * indent
        self.logger.info("%s- ❓ %s", spacer, node["question"])
        for child in node["children"]:
            self.print_tree(child, indent + 1)

    def print_answers(self, state: RAGState):
        self.logger.info("📜 Collected Q&A pairs:")
        for question, answer in state["answers"].items():
            self.logger.info("❓ Question: %s", question)
            self.logger.info("💡 Answer: %s", answer)
            self.logger.info("-" * 50)
    
    def save_graph_png(self, file_name = "graph_output.png"):
        with open(file_name, "wb") as f:
            f.write(self.app.get_graph().draw_mermaid_png())
            self.logger.info(f"graph mermaid png written → {file_name}")

    # ---------- Graph Nodes ----------
    def check_node(self, state: RAGState) -> RAGState:
        if not state["question_queue"]:
            self.logger.info("Question queue empty; skip check_node")
            return state

        current_query = state["question_queue"].pop(0)
        state["query"] = current_query
        node_depth = state["node_map"][current_query].get("depth", 1)
        self.logger.info("▶️ check: %s (depth=%d)", current_query, node_depth)
        entities = extract_entity(state["query"])
        triples = extract_triple(state["query"])

        bm25_docs = bm25_retrieve(
            query=state["query"], entities=entities, triples=triples, docs_set=self.docs_set, topk=self.topk
        )
        vector_docs = dense_retrieve(
            query=state["query"], entities=entities, triples=triples, docs_set=self.docs_set, topk=self.topk
        )

        query_et = query_with_et(query=state["query"], entities=entities, triples=triples)

        sub_qa = "\n".join(f"{k}: {v}" for k, v in state["answers"].items())

        self.logger.info(
            "retrieved: dense=%d, bm25=%d", len(vector_docs), len(bm25_docs)
        )

        status, provide_info = hybrid_response(
            state["query"], query_et, vector_docs, bm25_docs, k=self.topk, logger=self.logger, history_qa=sub_qa
        )
        
        state["node_map"][current_query]["provide_info"] = provide_info

        self.logger.info("❓❓❓❓queston: %s", current_query)
        if status["status"] == "sufficient":
            self.logger.info("✅✅✅✅ status: %s", status["status"])
            self.logger.info("✅✅✅✅ answer: %s", status["answer"])
            state["answers"][current_query] = status["answer"]
            state["node_map"][current_query]["answer"] = status["answer"]
            state["route_decision"] = "combine"
            self.logger.info("route → combine (answered)")
        elif status["status"] == "insufficient" and node_depth < self.max_depth:
            state["node_map"][current_query]["knowledge_gap"] = status["gap"]
            state["route_decision"] = "planner"
            self.logger.info("⚠️⚠️⚠️⚠️route → planner (insufficient)")
            self.logger.info("⚠️⚠️⚠️⚠️knowledge gap: %s", status["gap"])
        else:
            state["answers"][current_query] = "Insufficient information, unable to answer the current question."
            state["node_map"][current_query]["knowledge_gap"] = status["gap"]
            state["route_decision"] = "combine"
            self.logger.info("❗❗❗❗route → combine (max depth reached)")

        return state

    def planner_node(self, state: RAGState) -> RAGState:
        parent = state["node_map"][state["query"]]
        parent_depth = parent.get("depth", 1)

        parent_provide_info = parent.get("provide_info", "")

        parent_knowledge_gap = parent.get("knowledge_gap", "")

        suggestion_prefix = (
            f"Human Suggestion: {state['human_suggestion']}\n\n"
            if state.get("human_suggestion")
            else ""
        )

        sub_qa = "\n".join(f"{k}: {v}" for k, v in state["answers"].items())

        system_prompt = PROMPTS["DECOMPSITION_QUERY"].format(retrieved_chunks=parent_provide_info, history_qa=sub_qa, human_suggestion=suggestion_prefix, knowledge_gap=parent_knowledge_gap)

        self.logger.info("plan: %s", parent["question"])
        # raw = self.agent.chat(self.model, prompt, extra_body={"enable_thinking": False})
        raw = self.agent.chat(model=self.model, system_prompt=system_prompt, prompt=parent["question"])
        subqs = [q.strip() for q in raw.split("\n") if q.strip()]
        self.logger.info("planned subqs=%d", len(subqs))

        children: List[QuestionNode] = []
        to_enqueue: List[str] = []

        for subq in subqs:
            if subq in state["node_map"]:
                self.logger.info("skip duplicate subq: %s", subq)
                continue

            node: QuestionNode = {
                "id": str(uuid.uuid4()),
                "question": subq,
                "answer": None,
                "depth": parent_depth + 1,  
                "children": [],
            }
            children.append(node)
            to_enqueue.append(subq)
            state["node_map"][subq] = node
            self.logger.info("enqueue: %s", subq)
        
        if to_enqueue:
            # head insert
            state["question_queue"][0:0] = to_enqueue
            for q in to_enqueue:
                self.logger.info("enqueue (prepend): %s", q)
        parent["children"] = children
        return state

    def combine_node(self, state: RAGState) -> RAGState:
        if state["question_queue"]:
            state["route_decision"] = "check"
            self.logger.info(
                "combine postponed → back to check (queue size=%d)",
                len(state["question_queue"]),
            )
            return state

        root = state["root_query"]
        sub_qa = "\n".join(f"{k}: {v}" for k, v in state["answers"].items())
        suggestion_prefix = (
            f"Human Suggestion: {state['human_suggestion']}\n\n"
            if state.get("human_suggestion")
            else ""
        )
        system_prompt = (
            suggestion_prefix
            + f"""Given sub-QA pairs:
        {sub_qa}

        Synthesize answer strictly based on provided info for:
        Input Root Query:
        """
        )
        self.logger.info(f"combine prompt{system_prompt}")
        final_answer = self.agent.chat(model=self.model, system_prompt=system_prompt, prompt=root)
        state["answers"][root] = final_answer
        self.logger.info("final(head): %s", final_answer.replace("\n", " ")[:240])
        state["route_decision"] = "exit"
        self.logger.info(
                "combine → exit (queue size=%d)",
                len(state["question_queue"]),
            )
        return state

    def exit_node(self, state: RAGState) -> RAGState:
        self.logger.info("exit node reached, finishing graph execution")
        return state

    # ---------- Build Graph ----------
    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.set_entry_point("check")
        graph.add_node("check", self.check_node)
        graph.add_node("planner", self.planner_node)
        graph.add_node("combine", self.combine_node)
        graph.add_node("exit", self.exit_node)
        graph.add_conditional_edges(
            "check",
            lambda s: s["route_decision"],
            {"planner": "planner", "combine": "combine"},
        )
        graph.add_conditional_edges(
            "combine",
            lambda s: s["route_decision"],
            {"check": "check", "exit": "exit"},
        )
        graph.add_edge("planner", "combine")
        graph.add_edge("exit", END)
        return graph.compile(checkpointer=InMemorySaver())

    # ---------- Public API ----------
    def run(
        self,
        init_query: str
    ) -> RAGState:
        state = self.create_initial_state(init_query)
        self.logger.info("thread_id: %s", self.config["configurable"]["thread_id"])
        self.logger.info("run: %s", init_query)
        result = self.app.invoke(state, config=self.config)
        self.logger.info(
            "done: final answer length=%d", len(result["answers"].get(init_query, ""))
        )
        return result

if __name__ == "__main__":

    qa_path = "dataset/raw_QA.json"
    with open(qa_path) as f:
        qa_list = json.load(f)
    
    print(len(qa_list))

    data_choices = [2, 3, 4]

    chosen_qas = [qa_list[i] for i in data_choices]

    for idx, qa_pair in enumerate(chosen_qas):
        docs_set = f"crag_data{data_choices[idx]}"
        query = qa_pair["query"]
        answer = qa_pair["answer"]
        rag = MiniRAG(log_name=query, docs_set=docs_set)
        rag.run(query)
