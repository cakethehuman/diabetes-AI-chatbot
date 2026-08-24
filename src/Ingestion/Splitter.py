from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def splitDocuments(docs: list[Document]) -> list[Document]:
    logger.info("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    return splitter.split_documents(docs)



