# app/src/evaluation/faithfulness_evaluation.py
from ragas.metrics import faithfulness
from ragas.integrations.langchain import EvaluatorChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class FaithfulnessEvaluator:
    
    """" Does the generated answer accurately reflect the information in the provided contexts? """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        self.chain = EvaluatorChain(
            metric=faithfulness,
            llm=self.llm,
            embeddings=self.embeddings,
        )

    def evaluate(self, question: str, answer: str, contexts: list[str]) -> float:
        result = self.chain.invoke(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
            }
        )
        return result["faithfulness"]
