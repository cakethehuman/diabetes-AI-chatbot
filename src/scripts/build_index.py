from pathlib import Path
from src.utils.Settings import settings
from src.ingestion.Indexer import indexPath
from src.utils.logger import get_logger

import chromadb

logger = get_logger(__name__)
DATA_DIR = Path("data")

def is_store_empty() -> bool:
    client = chromadb.PersistentClient(path=str(settings.SEMSEARCH_DB))
    collections = client.list_collections()
    return not collections or all(c.count() == 0 for c in collections)

def main() -> None:
    if is_store_empty():
        logger.info("Vector store empty — indexing now...")
        count = indexPath(DATA_DIR)
        logger.info(f"Indexed {count} chunks")
    else:
        logger.info("Vector store already populated — skipping")

if __name__ == "__main__":
    main()