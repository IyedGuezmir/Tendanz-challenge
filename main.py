from app.src.retrieval import rag  # now works

def main():
    query_text = "Que se passe-t-il en cas de panne du véhicule ?"
    results = rag.query(query_text, k=4)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content[:500])

if __name__ == "__main__":
    main()
