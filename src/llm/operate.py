import json

import requests

from config.config import GLOABLE_CONFIG
from llm.agent import Agent
from llm.prompts import PROMPTS
from rag.utils import clean_json_text

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


def hybrid_response(query, vector_docs, bm25_docs, k=4, temperature=0.3):
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
    rerank_idx = rerank(query, total_docs, top_n=k)
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
        temperature=temperature,
    )

    response = clean_json_text(response)

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
    return answer


if __name__ == "__main__":
    from src.retriever.bm25_retriever import BM25Retriever
    from src.retriever.dense_retriever import DenseRetriever

    bm25 = BM25Retriever()
    vector = DenseRetriever()

    # test
    # query = "Who is the author of 'A Christmas Carol'?"

    # data0
    # query = "How many 3-point attempts did Steve Nash average per game in seasons he made the 50-40-90 club?"
    # query = "What is 3-point attempt?"
    # query = "In which seasons did Steve Nash achieve the 50-40-90 club?"
    # query = "What was Steve Nash's per game 3PA(3-point field goal attempts) in season 2005-06?"
    # query = "What was Steve Nash's per game 3PA(3-point field goal attempts) in season 2007-08?"
    # query = "What was Steve Nash's per game 3PA(3-point field goal attempts) in season 2008-09?"
    # query = "What was Steve Nash's per game 3PA(3-point field goal attempts) in season 2009-10?"
    # query = "What is the average 3-point attempts(3PA) per game of Steve Nash around season 2005-06, 2007-08, 2008-09, 2009-10?"

    # data1
    # query = "Are there any movies that feature a person who creates and controls a device?"
    # query = "what is a movie to feature a person who can create and control a device that can manipulate the laws of physics?"

    # data2
    # query = "where did the ceo of salesforce previously work?"
    # query = "who is the ceo of salesforce?"
    # query = "where did Marc Benioff previously work"

    # data3
    # query = "which movie won the oscar best visual effects in 2021?"

    # data4
    # query = "what company in the dow jones is the best performer today?"

    # data5
    query = "in 2004, which animated film was recognized with the best animated feature film oscar?"

    # data6
    # query = "on which date did sgml distribute dividends the first time"
    # query = "What is SGML from the provided infomation?"
    # query = "When did Sigma Lithium Corporation Common Shares (SGML) distribute dividends for the first time?"

    # data7
    # query = "what is the average gross for the top 3 pixar movies?"
    # query = "What are the top 3 Pixar movies by gross revenue in 2024?"

    # data8
    # query = "what are the countries that are located in southern africa."

    # data9
    # query = "which company in the s&p 500 index has the highest percentage of green energy usage?"
    # query = "which company has the highest percentage of green energy usage?"

    # data10
    # query = "what's the cooling source of the koeberg nuclear power station?"

    # data11
    # query = "when did hamburg become the biggest city of germany?"

    # data12
    # query = "how much did voyager therapeutics's stock rise in value over the past month?"
    query = (
        "how much did voyager therapeutics's stock change in value over the past month?"
    )

    # data13
    # query = ""

    data = "crag_data12"
    topk = 6

    vector_docs = vector.retrieve(query, collection_name=data, k=topk)
    bm25_docs = bm25.retrieve(query, index_name=data, k=topk)

    answer = hybrid_response(query, vector_docs, bm25_docs, k=topk, temperature=0.6)
    print(
        f"#############################\nQuery:\n{query}\n#############################\nAnswer:\n{answer}\n#############################"
    )
