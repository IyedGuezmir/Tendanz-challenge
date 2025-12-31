
from docling.document_converter import DocumentConverter
from pathlib import Path

def parse_pdf(input_doc_path: Path, output_dir: Path):

    md_filepath = output_dir / f"{input_doc_path.stem}-parsed-text.md"
    
    if md_filepath.exists():
        return md_filepath
    
    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(input_doc_path)

    with md_filepath.open("w", encoding="utf-8") as md_file:
        md_file.write(conv_res.document.export_to_markdown())
    
    return md_filepath