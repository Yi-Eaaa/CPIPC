import json

import requests
from config.config import GLOABLE_CONFIG

from llm.agent import Agent
from llm.prompts import PROMPTS

chat_key = GLOABLE_CONFIG["chat_api_key"]
rerank_key = GLOABLE_CONFIG["rerank_api_key"]


def rerank(
    query,
    documents,
    model="BAAI/bge-reranker-v2-m3",
    top_n=4,
    return_documents=False,
    max_chunks_per_doc=1024,
    overlap_tokens=128,
    with_score=False,
):
    url = GLOABLE_CONFIG["rerank_url"]

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": return_documents,
        "max_chunks_per_doc": max_chunks_per_doc,
        "overlap_tokens": overlap_tokens,
    }

    headers = {
        "Authorization": f"Bearer {rerank_key}",
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    response_json = json.loads(response.text)
    rank = []
    score = []
    for item in response_json["results"]:
        rank.append(item["index"])
        score.append(item["relevance_score"])
    if with_score:
        return rank, score
    else:
        return rank


def hybrid_response(query, vector_docs, bm25_docs):
    """
    Combine vector search and BM25 search results to generate a hybrid response.

    Args:
        query (str): The user's query.
        vector_docs (list): List of documents retrieved from vector search.
        bm25_docs (list): List of documents retrieved from BM25 search.

    Returns:
        str: A combined response from both search methods.
    """
    total_docs = vector_docs + bm25_docs
    rerank_idx = rerank(query, total_docs)
    provided_info = "#############################\n"
    for id, idx in enumerate(rerank_idx, start=1):
        provided_info += (
            f"Context {id}:\n\n{total_docs[idx]}\n#############################\n"
        )

    system_prompt = PROMPTS["RAG_PROMPT"].format(documents=provided_info)
    agent = Agent(api_key=chat_key)
    response = agent.chat(
        model=GLOABLE_CONFIG["chat_model"],
        prompt=query,
        system_prompt=system_prompt,
        extra_body={"enable_thinking": False},
    )

    try:
        response_json = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e} {response}.")

    answer = response_json["Answer"]
    ifsufficient = response_json["Ifsufficient"]

    if ifsufficient == "SUFFICIENT":
        return answer
    elif ifsufficient == "INSUFFICIENT":
        return "INSUFFICIENT."
    else:
        raise ValueError(
            f"Invalid response '{ifsufficient}' from LLM for 'Ifsufficient'."
        )


if __name__ == "__main__":
    query = "Who is the author of 'A Christmas Carol'?"
    from src.retriever.dense_retriever import DenseRetriever

    vector = DenseRetriever()
    vector_docs = vector.retrieve(query)

    from src.retriever.bm25_retriever import BM25Retriever

    bm25 = BM25Retriever()
    bm25_docs = bm25.retrieve(query)

    answer = hybrid_response(query, vector_docs, bm25_docs)
    print(
        f"#############################\nQuery:\n{query}\n#############################\nAnswer:\n{answer}\n#############################"
    )
