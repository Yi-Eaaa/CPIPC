import json
import os

import requests

from config.config import GLOABLE_CONFIG
from llm.agent import Agent
from llm.prompts import PROMPTS
from rag.utils import clean_json_text, process_html
from rag.text_to_triple import Text2Triple
from rag.text_to_entity import Text2Entities
from retriever.bm25_retriever import BM25Retriever
from retriever.dense_retriever import DenseRetriever

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


def hybrid_response(query, query_et, vector_docs, bm25_docs, k=4, temperature=0.3, history_qa = None, only_context = False, logger = None):
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
    rerank_idx = rerank(query_et, total_docs, top_n=k)
    provided_info = "#############################\n"
    for id, idx in enumerate(rerank_idx, start=1):
        provided_info += (
            f"Context {id}:\n\n{total_docs[idx]}\n#############################\n"
        )

    if only_context:
        return provided_info

    system_prompt = PROMPTS["RAG_PROMPT"].format(documents=provided_info, history_qa=history_qa)
    agent = Agent(api_key=chat_key)
    response = agent.chat(
        model=GLOABLE_CONFIG["chat_model"],
        prompt=query,
        system_prompt=system_prompt,
        # extra_body={"enable_thinking": False},
        temperature=temperature,
        logger=logger
    )

    response = clean_json_text(response)

    try:
        response_json = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e} {response}.")

    print(response)

    answer = response_json["Answer"]

    if answer == "Insufficient":
        knowledge_gap = response_json["Knowledge_gap"]
        return {"status": "insufficient", "gap": knowledge_gap}, provided_info
    
    return {"status": "sufficient", "answer": answer}, provided_info


def query_with_et(query, entities, triples, temperature=0.3):
    """
    Build a rewriting prompt that instructs the LLM to augment the query
    with entities and triples in natural language form.
    """

    system_prompt = PROMPTS["COMBINE_ENTITY_TRIPLE"].format(
        query=query,
        entities=json.dumps(entities, ensure_ascii=False),
        triples=json.dumps(triples, ensure_ascii=False),
    )
    agent = Agent(api_key=chat_key)
    response = agent.chat(
        model=GLOABLE_CONFIG["chat_model"],
        prompt=query,
        system_prompt=system_prompt,
        # extra_body={"enable_thinking": False},
        temperature=temperature,
    )

    response = clean_json_text(response)

    return response


def extract_entity(query):
    text2entities = Text2Entities()
    extracted_entities = text2entities.extract_entities(query, False)
    return extracted_entities


def extract_triple(query):
    text2triples = Text2Triple()
    extracted_triples = text2triples.generate_triple(query, True)
    return extracted_triples


def bm25_retrieve(query, entities, triples, docs_set, topk, with_score=False):
    bm25 = BM25Retriever()
    re_docs = {}

    query_docs = bm25.retrieve(
        query=query, index_name=docs_set, k=topk, with_score=with_score
    )
    for id, doc in query_docs.items():
        if id not in re_docs:
            re_docs[id] = doc

    for entity in entities:
        e_docs = bm25.retrieve(
            query=entity, index_name=docs_set, k=topk, with_score=with_score
        )
        for id, doc in e_docs.items():
            if id not in re_docs:
                re_docs[id] = doc

    for triple in triples:
        s_docs = bm25.retrieve(
            query=triple[0], index_name=docs_set, k=topk, with_score=with_score
        )
        p_docs = bm25.retrieve(
            query=triple[1], index_name=docs_set, k=topk, with_score=with_score
        )
        o_docs = bm25.retrieve(
            query=triple[2], index_name=docs_set, k=topk, with_score=with_score
        )
        for id, doc in s_docs.items():
            if id not in re_docs:
                re_docs[id] = doc
        for id, doc in p_docs.items():
            if id not in re_docs:
                re_docs[id] = doc
        for id, doc in o_docs.items():
            if id not in re_docs:
                re_docs[id] = doc

    re_ = list(re_docs.values())
    re_ = duplicate_docs(re_)
    return re_


def dense_retrieve(query, entities, triples, docs_set, topk, with_score=False):
    vector = DenseRetriever()
    re_docs = {}

    query_docs = vector.retrieve(
        query=query, collection_name=docs_set, k=topk, with_score=with_score
    )
    for id, doc in query_docs.items():
        if id not in re_docs:
            re_docs[id] = doc

    for entity in entities:
        e_docs = vector.retrieve(
            query=entity, collection_name=docs_set, k=topk, with_score=with_score
        )
        for id, doc in e_docs.items():
            if id not in re_docs:
                re_docs[id] = doc

    for triple in triples:
        s_docs = vector.retrieve(
            query=triple[0], collection_name=docs_set, k=topk, with_score=with_score
        )
        p_docs = vector.retrieve(
            query=triple[1], collection_name=docs_set, k=topk, with_score=with_score
        )
        o_docs = vector.retrieve(
            query=triple[2], collection_name=docs_set, k=topk, with_score=with_score
        )
        for id, doc in s_docs.items():
            if id not in re_docs:
                re_docs[id] = doc
        for id, doc in p_docs.items():
            if id not in re_docs:
                re_docs[id] = doc
        for id, doc in o_docs.items():
            if id not in re_docs:
                re_docs[id] = doc

    re_ = list(re_docs.values())
    re_ = duplicate_docs(re_)
    return re_


def duplicate_docs(docs):
    unique_docs = list(set(docs))
    return unique_docs


def test_data(data_choice, topk=10, temperature=0.3):
    base_dir = "/home/hdd1/QA-Dataset/CRAG-KDD-Cup-2024/crag-retrieval-summarization"
    raw_data_path = os.path.join(base_dir, "crag_task_1_dev_v4_release.jsonl")
    md_folder = os.path.join(base_dir, "md_data")
    os.makedirs(md_folder, exist_ok=True)

    select_data = None

    with open(raw_data_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i == data_choice:
                data = json.loads(line)
                if "query" in data and "answer" in data:
                    select_data = {
                        "query": data["query"],
                        "answer": data["answer"],
                        "page": [
                            process_html(page["page_result"])
                            for page in data["search_results"]
                        ],
                    }
                break

    choice_data_path = os.path.join(md_folder, f"data{data_choice}")
    os.makedirs(choice_data_path, exist_ok=True)

    for pid, page in enumerate(select_data["page"]):
        page_path = os.path.join(choice_data_path, f"page{pid}.md")
        with open(page_path, "w", encoding="utf-8") as f:
            f.write(page)

    bm25 = BM25Retriever()
    dense = DenseRetriever()

    docs_set = f"crag_data{data_choice}"

    bm25.construct_index(choice_data_path, index_name=docs_set)
    dense.construct_index(choice_data_path, collection_name=docs_set)

    query = select_data["query"]
    answer = select_data["answer"]

    entities = extract_entity(query)
    triples = extract_triple(query)

    bm25_docs = bm25_retrieve(
        query=query, entities=entities, triples=triples, docs_set=docs_set, topk=topk
    )
    vector_docs = dense_retrieve(
        query=query, entities=entities, triples=triples, docs_set=docs_set, topk=topk
    )

    query_et = query_with_et(query=query, entities=entities, triples=triples)

    status, provide_info = hybrid_response(
        query, query_et, vector_docs, bm25_docs, k=topk, temperature=temperature
    )
    print(
        f"#############################\nQuery:\n{query}\n#############################\nLLM Response:\n{status}\n#############################\nAnswer:\n{answer}\n#############################"
    )


if __name__ == "__main__":

    test_data(0)

    # bm25 = BM25Retriever()
    # vector = DenseRetriever()

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
    # query = "in 2004, which animated film was recognized with the best animated feature film oscar?"

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

    # data13
    # query = ""

    # data = "crag_data12"
    # topk = 10

    # entities = extract_entity(query)
    # triples = extract_triple(query)

    # bm25_docs = bm25_retrieve(
    #     query=query, entities=entities, triples=triples, docs_set=data, topk=topk
    # )
    # vector_docs = dense_retrieve(
    #     query=query, entities=entities, triples=triples, docs_set=data, topk=topk
    # )

    # query_et = query_with_et(query=query, entities=entities, triples=triples)

    # answer = hybrid_response(
    #     query, query_et, vector_docs, bm25_docs, k=topk, temperature=0.6
    # )
    # print(
    #     f"#############################\nQuery:\n{query}\n#############################\nAnswer:\n{answer}\n#############################"
    # )
    # pass
