from src.utils.logger import get_logger

from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader

logger = get_logger(__name__)

class Loader(BaseModel):
    links : list[str] | None = []

    def loadPdf(self) -> list:
        documents = []
        for url in self.links:
            logger.info(f"Loading pdf from url : {url}")
            loader = PyPDFLoader(url)
            documents.extend(loader.load())

        return documents
  
