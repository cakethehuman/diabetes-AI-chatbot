from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.ingestion.Indexer import getStore
from src.ingestion.Indexer import indexPath
from src.ingestion.Indexer import buildEmbeddings

from src.utils.logger import get_logger

logger = get_logger(__name__)

def search(query: str, k: int = 4) -> list[tuple[Document, float]]:
    query = query.strip()
    if not query:
        raise ValueError("query kosong")
    if k < 1:
        raise ValueError("k harus >= 1")

    store = getStore(buildEmbeddings())
    return store.similarity_search_with_score(query, k=k)


def get_retriever(k: int = 4):
    logger.info("Gettin top k similar data")
    return getStore(buildEmbeddings()).as_retriever(search_kwargs={"k": k})

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
            indexPath(args.path)
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