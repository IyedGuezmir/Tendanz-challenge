from app.src.retrieval import rag 
from app.src.retrieval import reranker
from app.src.generation import chain, decomp_chain
from app.src.generation import rag_fusion_chain
from app.src.evaluation.faithfulness_evaluation import FaithfulnessEvaluator
from app.src.evaluation.context_precision_evaluation import ContextPrecisionEvaluator
from app.utils.helpers import extract_ground_truth

def main():
    
    faithfulness_evaluator = FaithfulnessEvaluator()
    context_precision_evaluator = ContextPrecisionEvaluator()
    query_text = "Quelles sont les exclusions applicables ?"
    ground_truth = extract_ground_truth(query_text)  # reads from the default markdown file
 
    
    """ results = rag.query(query_text, k=4)
        top_results = reranker.rerank(query_text, results, top_k=4)


        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(doc.page_content[:500])
        
        print("\n=== After Reranking ===")
        for i, doc in enumerate(top_results, 1):
            print(f"\n--- Reranked Result {i} ---")
            print(doc.page_content[:500])
        """
    


    answer = chain.invoke(query_text)
    docs = chain._retrieve(query_text)
    contexts = [doc.page_content for doc in docs]
    print(answer)

    faithfulness_score = faithfulness_evaluator.evaluate(
    question=query_text,
    answer=answer,
    contexts=contexts,
   )
    context_precision_score = context_precision_evaluator.evaluate(
        question=query_text,
        ground_truth=ground_truth,
        contexts=contexts,
    )     
    print(f"Faithfulness Score: {faithfulness_score}")
    print(f"Context Precision Score: {context_precision_score}")
    #decomp_answer = decomp_chain.invoke(query_text)
    #print(decomp_answer)
    answer = rag_fusion_chain.answer(query_text)
    docs = rag_fusion_chain.retrieve(query_text, k_per_query=4)
    contexts = [doc.page_content for doc in docs]


    print(answer)
    faithfulness_score = faithfulness_evaluator.evaluate(
        question=query_text,
        answer=answer,
        contexts=contexts,
    )   
    context_precision_score = context_precision_evaluator.evaluate(
    question=query_text,
    ground_truth=ground_truth,
    contexts=contexts,
    )  
    print(f"Faithfulness Score: {faithfulness_score}")
    print(f"Context Precision Score: {context_precision_score}")


if __name__ == "__main__":
    main()
