#app/test/eval.py
from app.utils.helpers import prepare_qa_dict_from_text
from app.src.loader.load_chunk_docs import load_markdown_text
from dotenv import load_dotenv
load_dotenv()

md_text = load_markdown_text()
qa_dict = prepare_qa_dict_from_text(md_text)
print("CSV saved to app/data/eval/qa_dict.csv")

for q, a in qa_dict.items():
    print(f"Q: {q}\nA: {a}\n")
    
