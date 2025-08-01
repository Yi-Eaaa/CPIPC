import os
import sys
import uuid

from typing import Literal
from pydantic import BaseModel
from typing import List, Dict

from src.llm.api import SiliconFlowAPI
from src.config.config import GLOABLE_CONFIG
from src.llm.operate import hybrid_response
from src.retriever.dense_retriever import DenseRetriever
from src.retriever.bm25_retriever import BM25Retriever
from src.rag.logger import Logger

from langgraph.graph import END, StateGraph, START
from langgraph.types import Command, interrupt, Interrupt
from langgraph.checkpoint.memory import MemorySaver

# 设置项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# 初始化日志
logger = Logger(log_name="MiniRAG").get_logger()

# 初始化 LLM
API_KEY = GLOABLE_CONFIG["chat_api_key"]
MODEL = GLOABLE_CONFIG["chat_model"]
llm = SiliconFlowAPI(API_KEY)

# 定义状态
class RAGState(BaseModel):
    query: str
    subquestions: List[str] = []
    answers: Dict[str, str] = {}
    current_depth: int = 1
    route_decision: str = ""
    user_decision: str = ""
    retry_times: int = 0
    human_suggestion: str = ""

# HITL 节点
from langgraph.graph import END

# 检查节点
def check_node(state: RAGState) -> RAGState:
    logger.info(f"🔍 检查问题：{state.query}（深度 {state.current_depth}）")
    retriever_vector = DenseRetriever()
    retriever_bm25 = BM25Retriever()
    vector_docs = retriever_vector.retrieve(state.query, 1)
    bm25_docs = retriever_bm25.retrieve(state.query, 1)
    answer = hybrid_response(state.query, vector_docs, bm25_docs)

    logger.info(f"answer: {answer}")

    if answer == "INSUFFICIENT" or answer == "INSUFFICIENT.":
        logger.info(f"⚠️ 无法直接回答：{state.query}")
        state.answers[state.query] = ""
        if state.current_depth >= 3:
            logger.info(f"❌ 达到最大深度，无法回答：{state.query}")
            state.route_decision = "combine"
        else:
            logger.info(f"🔄 需要规划子问题：{state.query}")
            state.route_decision = "planner"
    else:
        logger.info(f"✅ 回答成功：{state.query} => {answer[:50]}...")
        state.answers[state.query] = answer
        state.route_decision = "combine"

    return state

def check_route(state: RAGState) -> str:
    logger.info(f"🚦 路由决策：{state.query} -> {state.route_decision}")
    return state.route_decision

# 规划子问题
def planner_node(state: RAGState) -> RAGState:
    logger.info(f"🧩 规划子问题：{state.query}（深度 {state.current_depth}）")
    if state.human_suggestion:
        prompt = f"用户建议：{state.human_suggestion}\n\n请将这个复杂问题拆解为多个简单子问题，每行一个：\n问题：{state.query}"
    else:
        prompt = f"请将下面这个复杂问题拆解成多个可回答的子问题，每行一个：\n问题：{state.query}"
    response = llm.chat(MODEL, prompt)
    subqs = [q.strip() for q in response.split("\n") if q.strip()]
    state.subquestions = subqs
    logger.info(f"📌 生成子问题：{subqs}")

    for subq in subqs:
        if subq not in state.answers:
            logger.info(f"🔁 递归处理子问题：{subq}")
            sub_state = RAGState(
                query=subq,
                subquestions=[],
                answers=state.answers,
                current_depth=state.current_depth + 1
            )

            config = {"configurable": {"thread_id": uuid.uuid4()}}
            result = app.invoke(sub_state, config=config)
            state_obj = RAGState(**result)
            state.answers.update(state_obj.answers)


    all_sub_answers = "\n".join(f"{k}: {v}" for k, v in state.answers.items() if k != state.query)
    prompt = f"以下是对每个子问题的回答，请整合成一个完整连贯的答案来回答：{state.query}\n\n{all_sub_answers}"
    combined = llm.chat(MODEL, prompt)
    logger.info(f"🧠 主问题答案：{state.query} => {combined}\n")
    logger.info("\n📚 子问题及回答：")
    for k, v in state_obj.answers.items():
        if k != query:
            logger.info(f"- {k}: {v}")
    state.answers[state.query] = combined
    return state

# Next steps after approval
def exit_node(state: RAGState) -> RAGState:
    return state

def clear_except_retry(state: RAGState, suggestion: str) -> dict:
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


def combine_node(state: RAGState) -> Command[Literal["check", "exit"]]:
    if state.current_depth == 1:
        logger.info(f"✅ 达到顶层问题，人工介入")
        user_decision = ""
        # 从用户输入决策
        while True:
            user_decision = input("👉 请输入 'approve' 或 'retry': ").strip().lower()
            if user_decision in ("approve", "retry"):
                break
            print("❗ 输入无效，请重新输入。")
        print(f"👉 用户输入为：{user_decision}")

        if user_decision == "approve":
            logger.info(f"✅ 用户批准，返回最终答案")
            return Command(goto="exit", update={"decision": "approve"})
        elif user_decision == "retry":
            suggestion = input("💡 请输入你对如何拆解这个问题的建议（可选）: ").strip()
            logger.info(f"🔄 用户选择重试，建议为：{suggestion}")
            return Command(goto="check", update=clear_except_retry(state, suggestion))
    else:
        logger.info(f"❌ 未达到顶层问题，无需人工介入")
        return state

# 构建ii
workflow = StateGraph(RAGState)
workflow.add_node("check", check_node)
workflow.add_node("planner", planner_node)
workflow.add_node("combine", combine_node)
workflow.add_node("exit", exit_node)

workflow.set_entry_point("check")
workflow.add_conditional_edges("check", check_route, {
    "planner": "planner",
    "combine": "combine"
})
workflow.add_edge("planner", "combine")
workflow.add_edge("exit", END)

# 启用 Checkpointer
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 可视化图结构
with open("graph_output.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())
logger.info("📈 流程图已保存为 'graph_output.png'")

if __name__ == "__main__":
    query = "Who are the protagonists in *A Christmas Carol* and *What I Worked On* respectively?"
    logger.info(f"\n🚀 启动主流程，问题为：{query}")
    init_state = RAGState(query=query)
    config = {"configurable": {"thread_id": uuid.uuid4()}}
    
    result = app.invoke(init_state, config=config)

    # 检查是否中断
    if isinstance(result, dict) and "__interrupt__" in result:
        interrupt_data = result.get("__interrupt__")
        print("🛑 中断信息：")
        print(interrupt_data)
        print("🧠 LLM 输出：", interrupt_data[0].value)

        # 从用户输入决策
        while True:
            user_decision = input("👉 请输入 'approve' 或 'reject': ").strip().lower()
            if user_decision in ("approve", "reject"):
                break
            print("❗ 输入无效，请重新输入。")

        # 恢复流程
        final_result = app.invoke(
            Command(resume=user_decision),
            config=config
        )
        print("\n✅ 最终结果：")
        print(final_result)
    else:
        print("✅ 没有中断，结果为：", result)

    state_obj = RAGState(** result)

    print("\n✅ 最终回答：")
    print(state_obj.answers.get(query))

    print("\n📚 子问题及回答：")
    for k, v in state_obj.answers.items():
        if k != query:
            print(f"- {k}: {v}")
