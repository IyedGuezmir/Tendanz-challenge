from pathlib import Path
from typing import List
import uuid
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from app.src.loader.load_chunk_docs import load_markdown_text, get_chunks_using_markers, split_chunk_with_langchain

load_dotenv()  

class HybridRAG:
    """
    Handles loading documents, splitting into chunks, creating a hybrid
    Qdrant vector store (dense + sparse), and querying it.
    """

    def __init__(
        self,
        collection_name: str = "my_documents4",
        chunk_size: int = 5000,
        chunk_overlap: int = 200,
        client_url: str = None,
        api_key: str = None,
        batch_size: int = 5,  
        timeout: int = 120, 
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size

        self.client = QdrantClient(url=client_url, api_key=api_key, timeout=timeout)

        self.dense_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

        self.qdrant: QdrantVectorStore = None

        self.all_chunks = self._load_and_split_chunks()

        # Auto-create collection if missing
        self._ensure_collection_exists()
        # Auto-build vector store if empty
        self._ensure_vector_store()

    def _load_and_split_chunks(self) -> List[str]:
        """Load markdown text and split into major + minor chunks."""
        md_txt = load_markdown_text()
        major_chunks = get_chunks_using_markers(md_txt)

        all_chunks = []
        for chunk in major_chunks:
            all_chunks.extend(
                split_chunk_with_langchain(
                    chunk, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
            )

        print(f"[HybridRAG] Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _ensure_collection_exists(self):
        """Create collection if it does not exist yet."""
        existing_collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=3072, distance=Distance.COSINE)},
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
            print(f"[HybridRAG] Collection '{self.collection_name}' created.")
        else:
            print(f"[HybridRAG] Collection '{self.collection_name}' already exists. Using it.")

    def _ensure_vector_store(self):
        """Initialize vector store and add documents only if empty."""
        self.qdrant = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )

        existing_count = self.client.count(collection_name=self.collection_name).count
        if existing_count > 0:
            print(f"[HybridRAG] Collection already has {existing_count} documents. Skipping embedding.")
            return

        # Prepare documents
        documents = [
            Document(page_content=chunk, metadata={"doc_id": str(uuid.uuid4())})
            for chunk in self.all_chunks
        ]
        uuids = [doc.metadata["doc_id"] for doc in documents]

        # Insert in batches to avoid timeout
        print(f"[HybridRAG] Adding {len(documents)} documents in batches of {self.batch_size}...")
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i+self.batch_size]
            batch_ids = uuids[i:i+self.batch_size]
            self.qdrant.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"[HybridRAG] Added documents {i+1} to {i+len(batch_docs)}")

        print(f"[HybridRAG] Vector store created with {len(documents)} documents")

    def query(self, query_text: str, k: int = 4) -> List[Document]:
        """Run a hybrid search query against the Qdrant store."""
        if not self.qdrant:
            raise ValueError("Vector store not initialized.")
        return self.qdrant.similarity_search(query_text, k=k)


