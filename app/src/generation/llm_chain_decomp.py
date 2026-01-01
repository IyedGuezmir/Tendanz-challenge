from operator import itemgetter
from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from app.src.retrieval.hybrid_rag import HybridRAG
from app.src.retrieval.cohere_reranker import CohereReranker


class LLMChainWithDecomposition:
    """
    Optional retrieval chain with query decomposition.
    - Breaks user query into sub-questions
    - Retrieves answers from HybridRAG
    - Optionally reranks answers using a Cohere reranker
    """

    def __init__(self, rag: HybridRAG, reranker: CohereReranker = None, llm_model: str = "gpt-3.5-turbo"):
        self.rag = rag
        self.reranker = reranker
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0)

        # Decomposition prompt: break query into sub-questions
        decomposition_template = """You are a helpful assistant that generates multiple sub-questions related to an input question.
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation.
        Generate multiple search queries (2) related to: {question}"""
        self.decomposition_prompt = ChatPromptTemplate.from_template(decomposition_template)
        self.decomposition_chain = self.decomposition_prompt | self.llm | StrOutputParser() | (lambda x: x.split("\n"))

        # Answer aggregation prompt: combines context and Q&A pairs to answer a sub-question
        aggregation_template = """
        Here is the question you need to answer:

        --- 
        {question}
        ---

        Here is any available background question + answer pairs:

        ---
        {q_a_pairs}
        ---

        Here is additional context relevant to the question: 

        ---
        {context}
        ---

        Use the above context and any background question + answer pairs to answer the question: {question}
        """
        self.aggregation_prompt = ChatPromptTemplate.from_template(aggregation_template)
        self.aggregation_chain = self.aggregation_prompt | self.llm | StrOutputParser()

    def format_qa_pair(self, question: str, answer: str) -> str:
        """Format Q/A pair for aggregation"""
        return f"Question: {question}\nAnswer: {answer}"

    def invoke(self, question: str, top_k: int = 2) -> str:
        """Run the decomposition → retrieval → aggregation workflow"""
        
        # Step 1: Decompose the query into sub-questions
        sub_questions = self.decomposition_chain.invoke({"question": question})

        # Step 2: Retrieve answers for each sub-question and build Q/A pairs
        q_a_pairs = ""
        for sub_q in sub_questions:
            docs = self.rag.query(sub_q, k=top_k)
            
            # Optional reranking
            if self.reranker:
                docs = self.reranker.rerank(sub_q, docs, top_k=top_k)
            
            context = "\n".join([d.page_content for d in docs])
            
            # Aggregate the answer using LLM + context
            answer = self.aggregation_chain.invoke({
                "question": sub_q,
                "q_a_pairs": q_a_pairs,
                "context": context
            })
            
            q_a_pairs += "\n---\n" + self.format_qa_pair(sub_q, answer)
        
        return q_a_pairs.strip()
