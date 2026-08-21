from src.utils.logger import get_logger

from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200

class Splitter(BaseModel):
    documents : list[str] | None = []
    chunk_size : int | None = CHUNK_SIZE
    chunk_overlap : int | None = CHUNK_OVERLAP
    
    
    def split(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunk = text_splitter.split_documents(self.documents)
        return chunk
    
    