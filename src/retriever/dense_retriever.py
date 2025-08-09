import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter, MarkdownElementNodeParser
from llama_index.core.schema import MetadataMode
from llama_index.core.utils import truncate_text
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.siliconflow import SiliconFlow
from llama_index.vector_stores.chroma import ChromaVectorStore
import os
import pickle

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

Settings.chunk_size = GLOABLE_CONFIG["chunk_size"]
Settings.chunk_overlap = GLOABLE_CONFIG["chunk_overlap"]


class DenseRetriever:
    def __init__(
        self,
        vectordb_dir="./chroma_db",
    ) -> None:
        self.vectordb_dir = vectordb_dir
        self.chroma_client = chromadb.PersistentClient(path=vectordb_dir)
        # self.node_parser = SentenceSplitter(
        #     chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap
        # )
        # self.node_parser = MarkdownElementNodeParser(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        self.node_parser = MarkdownElementNodeParser()

    def construct_index(self, docs_dir="./docs", collection_name="default"):
        documents = SimpleDirectoryReader(docs_dir).load_data()
        nodes_file = f"./bm25_persist/nodes_{collection_name}.pkl"
        if os.path.exists(nodes_file):
            nodes = pickle.load(open(nodes_file, "rb"))
        else:
            nodes = self.node_parser.get_nodes_from_documents(documents)
        if collection_name in self.chroma_client.list_collections():
            self.chroma_client.delete_collection(name=collection_name)
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

        vector_index.storage_context.persist(persist_dir=self.vectordb_dir)

    def retrieve(self, query, k=4, with_score=False, collection_name="default"):
        collection = self.chroma_client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_vector_store(
            storage_context=storage_context,
            vector_store=vector_store,
        )

        # nodes_file = f"./bm25_persist/nodes_{collection_name}.pkl"
        # nodes = pickle.load(open(nodes_file, "rb"))
        # base_nodes, objects = self.node_parser.get_nodes_and_objects(nodes)
        # vector_index = VectorStoreIndex(nodes=base_nodes+objects)

        retriever = vector_index.as_retriever(similarity_top_k=k)
        results = retriever.retrieve(query)
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
        
    def query(self, query, k=4, collection_name="default"):
        """
        Query the index with a given query string.
        """
        nodes_file = f"./bm25_persist/nodes_{collection_name}.pkl"
        nodes = pickle.load(open(nodes_file, "rb"))
        base_nodes, objects = self.node_parser.get_nodes_and_objects(nodes)
        vector_index = VectorStoreIndex(nodes=base_nodes+objects)
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=k,
            verbose=True
        )
        result = query_engine.query(query)
        return result


if __name__ == "__main__":
    dense = DenseRetriever()
    # dense.construct_index(docs_dir="./docs")
    # query = "Who is the author of 'A Christmas Carol'?"
    # documents, scores = dense.retrieve(query, with_score=True)

    # for document, score in zip(documents, scores):
    #     print(f"Content: {document}\nScore: {score}\n")

    # dense.construct_index(docs_dir="./datasets/crag-retrieval-summarization/first_20_data/markdown/data0", collection_name="crag_data0")
    # # query = "In which seasons did Steve Nash achieve the 50-40-90 club?"
    # query = "What was Steve Nash's average number of 3-point attempts per game in 2005-06 season?"
    # documents, scores = dense.retrieve(query, with_score=True, collection_name="crag_data0")
    # for document, score in zip(documents, scores):
    #     print(f"Content: {document}\nScore: {score}\n")

    # dense.construct_index(docs_dir="./datasets/crag-retrieval-summarization/first_20_data/markdown/data1", collection_name="crag_data1")
    # query = "Are there any movies that feature a person who creates and controls a device?"
    # documents, scores = dense.retrieve(query, with_score=True, collection_name="crag_data1")
    # for document, score in zip(documents, scores):
    #     print(f"Content: {document}\nScore: {score}\n")

    i = 2
    while i < 20:
        dense.construct_index(f"./datasets/crag-retrieval-summarization/first_20_data/markdown/data{i}", collection_name=f"crag_data{i}")
        i += 1
