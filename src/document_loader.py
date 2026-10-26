"""
Document loader module for extracting text and metadata from PDFs and text files.
"""
from typing import List
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> List[Document]:
    """Loads a PDF file and returns a list of Document objects with page metadata.
    
    Args:
        file_path: Path to the PDF file.
        
    Returns:
        List of langchain_core.documents.Document objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    reader = PdfReader(str(path))
    documents: List[Document] = []
    
    file_name = path.name

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Clean up whitespace and non-printable characters
        cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        
        if cleaned_text:
            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "source": file_name,
                    "page": idx + 1,
                    "total_pages": len(reader.pages),
                    "file_path": str(path.resolve()),
                }
            )
            documents.append(doc)
            
    return documents


def load_text_or_markdown(file_path: str | Path) -> List[Document]:
    """Loads a plain text or Markdown file and returns a Document object."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    return [
        Document(
            page_content=content,
            metadata={
                "source": path.name,
                "page": 1,
                "total_pages": 1,
                "file_path": str(path.resolve()),
            }
        )
    ]


def load_document(file_path: str | Path) -> List[Document]:
    """Auto-detects document format and loads it into a list of Documents."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in [".txt", ".md"]:
        return load_text_or_markdown(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats are .pdf, .txt, .md")
