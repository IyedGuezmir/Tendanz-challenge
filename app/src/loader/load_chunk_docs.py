#app/src/loader/load_chunk_docs.py
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ...utils.helpers import parse_pdf


def _initialize_markdown(pdf_name: str = "cg-auto-test.pdf"):
    data_folder = Path("app/data/pdfs")
    output_dir = Path("app/data/parsed")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_doc_path = data_folder / pdf_name
    return parse_pdf(input_doc_path, output_dir)



def load_markdown_text() -> str:
    """Load the markdown file, initializing it if needed."""
    md_filepath = _initialize_markdown()
    return md_filepath.read_text(encoding="utf-8")

def get_chunks_using_markers(src_text: str) -> list[str]:
    """
    Split the source text into chunks using major section markers.
    """
    major_sections = [
        "# QUELQUES DÉFINITIONS",
        "# VOUS ET VOTRE CONTRAT",
        "# SYNTHÈSE DES GARANTIES PROPOSÉES",
        "# PRÉSENTATION DES GARANTIES",
        "# CE QUE VOTRE CONTRAT NE GARANTIT JAMAIS",
        "# EN CAS DE SINISTRE",
        "# LA VIE DE VOTRE CONTRAT",
        "# LES DÉCLARATIONS QUE VOUS DEVEZ FAIRE",
        "# LE KILOMÉTRAGE",
        "# LA COTISATION",
        "# VOTRE INFORMATION",
        "# FICHE D'INFORMATION"
    ]
    
    chunks = []
    current_pos = 0
    
    for section in major_sections:
        idx = src_text.find(section, current_pos)
        if idx != -1:
            if idx > current_pos:
                chunk = src_text[current_pos:idx].strip()
                if chunk:
                    chunks.append(chunk)
            current_pos = idx
    
    if current_pos < len(src_text):
        final_chunk = src_text[current_pos:].strip()
        if final_chunk:
            chunks.append(final_chunk)
    
    if not chunks:
        marker = "\n#"
        parts = src_text.split(marker)
        
        if parts[0].strip():
            chunks.append(parts[0].strip())
        
        for part in parts[1:]:
            if part.strip():
                chunks.append(marker + part.strip())
    
    return chunks


def split_chunk_with_langchain(chunk: str, chunk_size: int = 5000, chunk_overlap: int = 200) -> list[str]:
    """
    Split chunk while keeping tables and their explanations together.
    Only split at subheadings or if chunk is very large.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n##"],  # only split by minor headings
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(chunk)
    
    final_chunks = []
    for c in chunks:
        if final_chunks and len(c) < 300:  
            final_chunks[-1] += "\n\n" + c
        else:
            final_chunks.append(c)
    
    return final_chunks

