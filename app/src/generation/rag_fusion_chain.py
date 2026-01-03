# app/src/generation/rag_fusion_chain.py
from operator import itemgetter
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from langchain.schema import Document

from app.src.retrieval.hybrid_rag import HybridRAG


class RAGFusionChain:
    """
    Implements a RAG-Fusion workflow with optional final answer generation.
    - Generates multiple related queries from a single input
    - Retrieves documents for each query using Hybrid RAG
    - Fuses results using Reciprocal Rank Fusion
    - Optionally generates a final summarized answer using an LLM
    """

    def __init__(self, rag: HybridRAG, llm_model: str = "gpt-4o-mini", temp: float = 0.0):
        self.rag = rag
        self.llm = ChatOpenAI(model_name=llm_model, temperature=temp, streaming=True)

        template_queries = """ 
        You are a legal information retrieval assistant.
        Your task is to generate multiple search queries that retrieve legal texts
        relevant to the same legal question.

        Rules:
        - Preserve the original legal meaning exactly
        - Do NOT change legal scope or assumptions
        - Do NOT introduce new legal concepts
        - Do NOT ask hypothetical or advisory questions
        - Use formal legal language

        Return the original question followed by 2 more generated search queries.

        
        Original Question:
        {question}"""
        
        
        self.prompt_gen_queries = ChatPromptTemplate.from_template(template_queries)

        template_answer = """You are a helpful assistant. Answer the question based on the following context
        If you are unsure of the answer, reply I don't know. 
        Context:
        {context}

        Question: {question}

        """
        self.prompt_answer = ChatPromptTemplate.from_template(template_answer)

    def generate_queries(self, question: str) -> List[str]:
        """Generate multiple related queries for RAG-Fusion."""
        response = self.llm.invoke(
            self.prompt_gen_queries.format_prompt(question=question).to_messages()
        )
        
        # Extract the text content from AIMessage
        text = response.content  
        
        # Split into individual queries
        queries = [q.strip() for q in text.split("\n") if q.strip()]
        print(queries)
        return queries

    @staticmethod
    def reciprocal_rank_fusion(results: List[List[Document]], k: int = 60) -> List[Document]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
        fused_scores = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                fused_scores[doc_str] = fused_scores.get(doc_str, 0) + 1 / (rank + k)

        reranked = [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
        return [doc for doc, _ in reranked]

    def retrieve(self, question: str, k_per_query: int = 5) -> List[Document]:
        """Run RAG-Fusion retrieval for a given question."""
        queries = self.generate_queries(question)
        all_results = []
        for q in queries:
            results = self.rag.query(q, k=k_per_query)
            all_results.append(results)
        return self.reciprocal_rank_fusion(all_results)

    def answer(self, question: str, k_per_query: int = 5, top_k_docs: int = 6) -> str:
        """
        Retrieve RRF-fused documents and produce a final answer.
        - Uses only top_k_docs documents for context.
        - Does NOT truncate any text.
        """
        docs = self.retrieve(question, k_per_query=k_per_query)

        docs_to_use = docs[:top_k_docs]

        context = "\n\n".join([doc.page_content for doc in docs_to_use])

        response = self.llm.invoke(
            self.prompt_answer.format_prompt(context=context, question=question).to_messages()
        )
        text = response.content  
        return StrOutputParser().parse(text)

