from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.src.retrieval.hybrid_rag import HybridRAG
from app.src.retrieval.cohere_reranker import CohereReranker


class LLMChain:
    """
    End-to-end RAG chain:
    Query -> Hybrid Retrieval -> (Optional) Reranking -> LLM Generation
    """

    def __init__(
        self,
        rag: HybridRAG,
        reranker: Optional[CohereReranker] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        retrieve_k: int = 8,
        final_k: int = 4,
    ):
        self.rag = rag
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.final_k = final_k

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
Use only the context below to answer the question.
If the answer is not contained in the context, say You don't know.

Context:
{context}

Question:
{question}
"""
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def _retrieve(self, question: str) -> List[Document]:
        """Hybrid retrieval from Qdrant"""
        return self.rag.query(question, k=self.retrieve_k)

    def _rerank(
        self, question: str, documents: List[Document]
    ) -> List[Document]:
        """Optional Cohere reranking"""
        if not self.reranker:
            return documents[: self.final_k]

        return self.reranker.rerank(
            query=question,
            documents=documents,
            top_k=self.final_k,
        )

    @staticmethod
    def _build_context(documents: List[Document]) -> str:
        """Join documents into a single context string"""
        return "\n\n".join(doc.page_content for doc in documents)

    def invoke(self, question: str) -> str:
        """
        Full RAG pipeline:
        - retrieve
        - rerank (optional)
        - generate answer
        """

        retrieved_docs = self._retrieve(question)
        final_docs = self._rerank(question, retrieved_docs)

        context = self._build_context(final_docs)

        return self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )
