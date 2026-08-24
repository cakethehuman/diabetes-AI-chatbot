from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from src.rag.JinaEmbeddings import JinaEmbeddings

from src.utils.Settings import settings

PERSIST_DIR = settings.SEMSEARCH_DB
COLLECTION = settings.SEMSEARCH_COLLECTION
JINA_MODEL = settings.JINA_MODEL
JINA_DIMENSIONS = settings.JINA_DIMENSIONS
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}

def build_embeddings() -> Embeddings:
    return JinaEmbeddings(model=JINA_MODEL, dimensions=JINA_DIMENSIONS)

def load_documents(source: Path) -> list[Document]:
    """Baca satu file atau semua file yang didukung di dalam folder."""
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


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(docs)

def get_store(embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

def index_path(source: Path) -> int:
    """Baca dokumen dari source, chunk, embed, simpan. Return jumlah chunk."""
    docs = load_documents(source)
    chunks = split_documents(docs)
    if not chunks:
        raise ValueError("tidak ada chunk yang dihasilkan — cek isi file")

    store = get_store(build_embeddings())
    store.add_documents(chunks)
    print(f"{len(docs)} dokumen -> {len(chunks)} chunk, tersimpan di {PERSIST_DIR}")
    return len(chunks)


def search(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Cari chunk paling relevan. Return list (Document, score)."""
    query = query.strip()
    if not query:
        raise ValueError("query kosong")
    if k < 1:
        raise ValueError("k harus >= 1")

    store = get_store(build_embeddings())
    return store.similarity_search_with_score(query, k=k)


def get_retriever(k: int = 4):
    """Retriever buat dipasang ke chain / agent RAG."""
    return get_store(build_embeddings()).as_retriever(search_kwargs={"k": k})

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LangChain semantic search (embedding: Jina AI)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="index file atau folder")
    p_index.add_argument("path", type=Path)

    p_search = sub.add_parser("search", help="cari di index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=4, help="jumlah hasil (default 4)")

    args = parser.parse_args(argv)

    try:
        if args.cmd == "index":
            index_path(args.path)
        else:
            hits = search(args.query, args.k)
            if not hits:
                print("tidak ada hasil — sudah di-index belum?")
                return 1
            for rank, (doc, score) in enumerate(hits, start=1):
                src = doc.metadata.get("source", "?")
                page = doc.metadata.get("page")
                loc = f"{src}" + (f" (hal. {page})" if page is not None else "")
                preview = " ".join(doc.page_content.split())[:300]
                print(f"\n#{rank}  score={score:.4f}  {loc}\n{preview}")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())