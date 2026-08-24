import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}

def loadDocuments(source: Path) -> list[Document]:
    logger.info("Loading documents...")
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"path tidak ada: {source}")

    files = [source] if source.is_file() else sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(f"tidak ada file {sorted(SUPPORTED_SUFFIXES)} di {source}")

    docs: list[Document] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            print(f"[skip] format tidak didukung: {path}", file=sys.stderr)
            continue

        if suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            docs.append(Document(page_content=text, metadata={"source": str(path)}))

    return docs