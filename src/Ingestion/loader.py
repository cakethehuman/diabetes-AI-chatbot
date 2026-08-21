from src.utils.logger import get_logger

from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader

logger = get_logger(__name__)

LINKS = [
    "https://med.virginia.edu/family-medicine/wp-content/uploads/sites/285/2021/06/PE07032_eng_Living-With-Diabetes-ACP-1.pdf",
    "https://www.cdc.gov/diabetes/pdfs/prevent/On-your-way-to-preventing-type-2-diabetes.pdf",
    "https://www.cardi-oh.org/files/resources/cardi-oh-lifestyle-changes-to-prevent-diabetes.pdf",
    "https://www.knowledge-action-portal.com/sites/default/files/2026-04/IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf"
]

class Loader(BaseModel):
    links : list[str] | None = []
    
    def clean_text(self, documents):
        clean_documents = []
        for doc in documents:
            raw = doc.page_content
            cleaned_text = " ".join(raw.split())
            clean_documents.append(cleaned_text)
        return clean_documents

    
    def loadPdf(self) -> list:
        all_cleaned_content = []
        for url in self.links:
            logger.info(f"Loading pdf from url : {url}")
            loader = PyPDFLoader(url)
            loaded_docs = loader.load()
            clean = self.clean_text(loaded_docs)
            all_cleaned_content.extend(clean)

        return all_cleaned_content
  
