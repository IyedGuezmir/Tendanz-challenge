import os
from dotenv import load_dotenv
from .hybrid_rag import HybridRAG 
from .cohere_reranker import CohereReranker
load_dotenv()

DEFAULT_COLLECTION = "my_documents4"

rag = HybridRAG(
    collection_name=DEFAULT_COLLECTION,
    client_url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
)
reranker = CohereReranker(api_key=os.environ.get("COHERE_API_KEY"))
