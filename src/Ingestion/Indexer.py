from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src.rag.JinaEmbeddings import JinaEmbeddings
from src.ingestion.Loader import loadDocuments
from src.ingestion.Splitter import splitDocuments
from src.utils.Settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

JINA_MODEL = settings.JINA_MODEL
JINA_DIMENSIONS = settings.JINA_DIMENSIONS
PERSIST_DIR = settings.SEMSEARCH_DB
COLLECTION = settings.SEMSEARCH_COLLECTION


def buildEmbeddings() -> Embeddings:
    return JinaEmbeddings(model=JINA_MODEL, dimensions=JINA_DIMENSIONS)    

def getStore(embeddings: Embeddings) -> Chroma:
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    return vectorstore
    
def indexPath(source: Path) -> int:
    docs = loadDocuments(source)
    chunks = splitDocuments(docs)
    if not chunks:
        raise ValueError("tidak ada chunk yang dihasilkan — cek isi file")
    store = getStore(buildEmbeddings())
    logger.info("Loading Chunks to vector store...")
    store.add_documents(chunks)
    print(f"{len(docs)} dokumen -> {len(chunks)} chunk, tersimpan di {PERSIST_DIR}")
    return len(chunks)
