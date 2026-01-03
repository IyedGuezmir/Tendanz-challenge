# app/src/evaluation/context_precision_evaluation.py
from ragas.metrics import context_precision
from ragas.integrations.langchain import EvaluatorChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class ContextPrecisionEvaluator:
    
    """" How Relevant are the provided contexts to answer the question? """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        self.chain = EvaluatorChain(
            metric=context_precision,
            llm=self.llm,
            embeddings=self.embeddings,
        )

    def evaluate(self, question: str, ground_truth: str, contexts: list[str]) -> float:
        result = self.chain.invoke(
            {
                "question": question,
                "ground_truth": ground_truth,
                "contexts": contexts,
            }
        )
        return result["context_precision"]