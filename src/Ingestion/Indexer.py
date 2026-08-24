from typing import Any

from langchain_chroma import Chroma

from pydantic import BaseModel

class Indexer(BaseModel):
    chunks : list[Any] | None = None    
    emebed_model : str | None = None

    def load_embeddings(chunks, embed_model) -> Chroma:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embed_model,
            persist_directory="./chroma_db",
        )
        return vectorstore