from app.src.retrieval import rag 
from app.src.retrieval import reranker

def main():

    query_text = "Que se passe-t-il en cas de panne du véhicule ?"
    results = rag.query(query_text, k=4)
    top_results = reranker.rerank(query_text, results, top_k=4)


    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:500])
    
    print("\n=== After Reranking ===")
    for i, doc in enumerate(top_results, 1):
        print(f"\n--- Reranked Result {i} ---")
        print(doc.page_content[:500])

if __name__ == "__main__":
    main()
