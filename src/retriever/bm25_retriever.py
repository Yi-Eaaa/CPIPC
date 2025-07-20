import os
import shutil

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.core.utils import truncate_text
from llama_index.retrievers.bm25 import BM25Retriever as BM25


class BM25Retriever:
    def __init__(
        self,
        chunk_size=512,
    ):
        self.text_splitter = SentenceSplitter(chunk_size=chunk_size)
        self.retriever = None

    def construct_index(self, docs_dir="./docs", persist_dir="./bm25_persist", k=4):
        """
        Create a new BM25 retriever and add documents to the index.
        """
        documents = SimpleDirectoryReader(docs_dir).load_data()
        nodes = self.text_splitter.get_nodes_from_documents(
            documents, show_progress=True
        )

        self.retriever = BM25.from_defaults(
            nodes=nodes,
            similarity_top_k=k,
            language="english",
        )
        self.persist(persist_dir)

    def existed_index(self, persist_dir="./bm25_persist"):
        self.retriever = BM25.from_persist_dir(persist_dir)

    def persist(self, persist_dir="./bm25_persist"):
        """
        Persist the BM25 index to the specified directory.
        """
        assert persist_dir, "Persist directory must be specified."
        if os.path.exists(persist_dir):
            print(f"Persist directory {persist_dir} already exists. Removing it.")
            shutil.rmtree(persist_dir)
        if self.retriever is None:
            raise ValueError(
                "Retriever has not been initialized. Call new_retriever or existing_retriever first."
            )

        self.retriever.persist(persist_dir)
        # print(f"BM25 index saved to {persist_dir} successfully.")

    def retrieve(self, query, with_score=False):
        """
        Retrieve the top-k documents for the given query using BM25.
        """
        if self.retriever is None:
            self.existed_index()
        results = self.retriever.retrieve(query)
        documents = []
        scores = []
        for node in results:
            source_text_fmt = truncate_text(
                node.node.get_content(metadata_mode=MetadataMode.NONE).strip(),
                max_length=5000,
            )
            documents.append(source_text_fmt)
            scores.append(node.score)

        if with_score:
            return documents, scores
        else:
            return documents


if __name__ == "__main__":
    bm25 = BM25Retriever()

    bm25.construct_index("./docs")
    query = "Who is the author of 'A Christmas Carol'?"
    documents, scores = bm25.retrieve(query, with_score=True)
    for document, score in zip(documents, scores):
        print(f"Content: {document}\nScore: {score}\n")

    # bm25.existed_index("./bm25_persist")
    # query = "Who is the author of 'A Christmas Carol'?"
    # documents, scores = bm25.retrieve(query, with_score=True)
    # for document, score in zip(documents, scores):
    #     print(f"Content: {document}\nScore: {score}\n")
