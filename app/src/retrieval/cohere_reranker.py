# app/src/retrieval/cohere_reranker.py
import os
import cohere
from typing import List
from langchain.schema import Document

class CohereReranker:
    def __init__(self, api_key: str = None):
        self.client = cohere.ClientV2()
        

    def rerank(self, query: str, documents: List[Document], top_k: int = 6) -> List[Document]:
        """
        documents: list of Document objects with 'page_content'
        Returns: reranked list of documents
        """
        texts = [doc.page_content for doc in documents]
        response = self.client.rerank(
            model='rerank-v3.5',
            query=query,
            documents=texts
        )
        ranked_docs = [documents[i.index] for i in response.results]
        return ranked_docs[:top_k]