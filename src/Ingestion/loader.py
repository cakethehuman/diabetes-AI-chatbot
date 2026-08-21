from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader

LINKS = [
    "https://med.virginia.edu/family-medicine/wp-content/uploads/sites/285/2021/06/PE07032_eng_Living-With-Diabetes-ACP-1.pdf",
    "https://www.cdc.gov/diabetes/pdfs/prevent/On-your-way-to-preventing-type-2-diabetes.pdf",
    "https://www.cardi-oh.org/files/resources/cardi-oh-lifestyle-changes-to-prevent-diabetes.pdf",
    "https://www.knowledge-action-portal.com/sites/default/files/2026-04/IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf"
]

class Loader(BaseModel):
    links : list[str] = []
    
    def loadPdf(self) -> list:
        documents = []

        for url in self.links:
            print("Loading")
            loader = PyPDFLoader(url)
            documents.extend(loader.load())

        return documents
    
test = Loader(links=LINKS)
test.loadPdf()