from pydantic import BaseModel, ConfigDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

class Splitter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    documents : list[Document] | None = []
    chunk_size : int | None = CHUNK_SIZE
    chunk_overlap : int | None = CHUNK_OVERLAP
    
    
    def split(self) -> list[Document]:
        logger.info("Splitting documents to chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunk = text_splitter.split_documents(self.documents)
        return chunk
    
