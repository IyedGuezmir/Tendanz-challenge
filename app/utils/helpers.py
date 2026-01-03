
from docling.document_converter import DocumentConverter
from pathlib import Path
import csv
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def parse_pdf(input_doc_path: Path, output_dir: Path):

    md_filepath = output_dir / f"{input_doc_path.stem}-parsed-text.md"
    
    if md_filepath.exists():
        return md_filepath
    
    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(input_doc_path)

    with md_filepath.open("w", encoding="utf-8") as md_file:
        md_file.write(conv_res.document.export_to_markdown())
    
    return md_filepath


def extract_ground_truth(query: str, md_file_path: str = "app/data/parsed/cg-auto-test-parsed-text.md") -> str:
    """
    Extract the minimal context from a markdown file needed to answer the query.
    Returns a string suitable as ground_truth for context_precision evaluation.
    """

    # Read the markdown file
    markdown_text = Path(md_file_path).read_text(encoding="utf-8")

    template = """You are an assistant that extracts only the text relevant to answer a question.
Do not add extra explanations, summaries, or unrelated content.

Question: {query}

Reference Text:
{markdown_text}

Extract only the portions of the text that are strictly needed to answer the question."""

    prompt = ChatPromptTemplate.from_template(template)

    response = llm.invoke(
        prompt.format_prompt(query=query, markdown_text=markdown_text).to_messages()
    )
    response = response.content

    return StrOutputParser().parse(response)

def prepare_qa_dict_from_text(
    text: str,
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    save_csv_path: Path = Path("app/data/eval/qa_dict.csv")
):
    """
    Extracts example Q&A from text using an LLM and saves as CSV.
    Each row: {"question": "...", "answer": "..."}
    """

    llm = ChatOpenAI(model_name=llm_model, temperature=temperature)

    # Define parser
    parser = SimpleJsonOutputParser()

    # Prompt
    prompt_template = (
        "You are an expert tutor. Extract 5-10 questions and their answers "
        "diversify the questions from tables contents to texts and concepts."
        "from the following text. Return JSON as a list of objects with 'question' and 'answer' keys.\n\n"
        "Text:\n{text}"
    )

    prompt = ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(prompt_template)
    ])

    # Get LLM output
    response = llm(prompt.format_prompt(text=text).to_messages())

    # Parse JSON
    try:
        qa_list = parser.parse(response.content)
    except Exception:
        print("Warning: Could not parse LLM output as JSON. Returning raw text.")
        qa_list = [{"question": "raw_output", "answer": response.content}]

    # Save CSV
    save_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with save_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(qa_list)

    # Return dictionary {question: answer}
    return {item["question"]: item["answer"] for item in qa_list}
