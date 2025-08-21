import os
import pickle
import shutil

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import (
    MarkdownElementNodeParser,
    SentenceSplitter,
    UnstructuredElementNodeParser,
)
from llama_index.core.schema import MetadataMode
from llama_index.core.utils import truncate_text
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.siliconflow import SiliconFlow
from llama_index.retrievers.bm25 import BM25Retriever as BM25

from config.config import GLOABLE_CONFIG

Settings.embed_model = OllamaEmbedding(
    model_name=GLOABLE_CONFIG["embedding_model"],
)

Settings.llm = SiliconFlow(
    # model=GLOABLE_CONFIG["chat_model"],
    model="deepseek-ai/DeepSeek-V3",
    api_key=GLOABLE_CONFIG["chat_api_key"],
    extra_body={"enable_thinking": False},
)
# print(Settings.llm.complete("Introduce yourself."))


class BM25Retriever:
    def __init__(
        self,
    ):
        self.node_parser = MarkdownElementNodeParser(num_workers=1)
        self.retriever = None
        self.base_dir = "./datasets/bm25_persist"

    def construct_index(self, docs_dir="./docs", index_name="test", k=4, overwrite=False):
        """
        Create a new BM25 retriever and add documents to the index.
        """
        nodes_path = self.base_dir
        os.makedirs(nodes_path, exist_ok=True)
        nodes_file = f"nodes_{index_name}.pkl"
        nodes_file = os.path.join(nodes_path, nodes_file)
        if os.path.exists(nodes_file) and not overwrite:
            nodes = pickle.load(open(nodes_file, "rb"))
        else:
            documents = SimpleDirectoryReader(docs_dir).load_data()
            nodes = self.node_parser.get_nodes_from_documents(documents, show_progress=True)
            pickle.dump(nodes, open(nodes_file, "wb"))

        self.retriever = BM25.from_defaults(
            nodes=nodes,
            similarity_top_k=k,
            language="english",
        )
        self.persist(index_name)

    def existed_index(self, index_name="test"):
        persist_dir = os.path.join(self.base_dir, index_name)
        self.retriever = BM25.from_persist_dir(persist_dir)
        # nodes = self.retriever.corpus
        # table = []
        # for node in nodes:
        #     if 'table_df' in node.keys():
        #         table.append({'table': node['table_df'], 'summary': node['table_summary']})
        # pass

    def persist(self, index_name="test"):
        """
        Persist the BM25 index to the specified directory.
        """
        persist_dir = os.path.join(self.base_dir, index_name)
        if os.path.exists(persist_dir):
            print(f"Persist directory {persist_dir} already exists. Removing it.")
            shutil.rmtree(persist_dir)
        if self.retriever is None:
            raise ValueError(
                "Retriever has not been initialized. Call new_retriever or existing_retriever first."
            )

        self.retriever.persist(persist_dir)
        # print(f"BM25 index saved to {persist_dir} successfully.")

    def retrieve(self, query, with_score=False, index_name="test", k=4):
        """
        Retrieve the top-k documents for the given query using BM25.
        """
        # if self.retriever is None:
        #     self.existed_index()
        self.existed_index(index_name)
        self.retriever.similarity_top_k = k
        results = self.retriever.retrieve(query)
        documents = {}
        scores = []
        for node in results:
            source_text_fmt = truncate_text(
                node.node.get_content(metadata_mode=MetadataMode.NONE).strip(),
                max_length=5000,
            )
            documents[node.node.id_] = source_text_fmt
            scores.append(node.score)

        if with_score:
            return documents, scores
        
        return documents


if __name__ == "__main__":
    bm25 = BM25Retriever()

    # # bm25.construct_index("./docs")
    # # query = "Who is the author of 'A Christmas Carol'?"
    # # documents, scores = bm25.retrieve(query, with_score=True)
    # # for document, score in zip(documents, scores):
    # #     print(f"Content: {document}\nScore: {score}\n")

    # bm25.construct_index("./datasets/crag-retrieval-summarization/first_20_data/markdown/data0", index_name="crag_data0")
    # # bm25.existed_index(index_name="crag_data0")
    # query = "In which seasons did Steve Nash achieve the 50-40-90 club?"
    # documents, scores = bm25.retrieve(query, with_score=True, index_name="crag_data0")
    # for document, score in zip(documents, scores):
    #     print(f"Content: {document}\nScore: {score}\n")

    # bm25.construct_index("./datasets/crag-retrieval-summarization/first_20_data/markdown/data1", index_name="crag_data1")
    # query = "Are there any movies that feature a person who creates and controls a device?"
    # documents, scores = bm25.retrieve(query, with_score=True, index_name="crag_data1")
    # for document, score in zip(documents, scores):
    #     print(f"Content: {document}\nScore: {score}\n")

    i = 0
    while i < 20:
        bm25.construct_index(
            f"./datasets/crag-retrieval-summarization/first_20_data/markdown/data{i}",
            index_name=f"crag_data{i}",
        )
        i += 1
