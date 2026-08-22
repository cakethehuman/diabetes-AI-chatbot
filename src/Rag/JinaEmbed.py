from __future__ import annotations

import os 
import time
from typing import Any
import requests

from langchain_core.embeddings import Embeddings

from pydantic import BaseModel, model_validator, PrivateAttr

API_URL = "https://api.jina.ai/v1/embeddings"
DEFAULT_MODEL = "jina-embeddings-v3"
DEFAULT_DIMENSIONS = 1024 
BATCH_SIZE = 64
TIMEOUT = 60
MAX_RETRIES = 3

class JinaEmbeddings(BaseModel, Embeddings):
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    dimensions: int | None = DEFAULT_DIMENSIONS
    late_chunking: bool = False
    batch_size: int = BATCH_SIZE
    _session: requests.Session = PrivateAttr(default_factory=requests.Session)
    
    @model_validator(mode='after')
    def verify_data(self):
        key = self.api_key or os.getenv('JINA_API_KEY')
        if not key:
            raise RuntimeWarning(
                "JINA_API_KEY is missing"                 
            )
        if not self.model:
            raise RuntimeWarning(
                "model is missing"                 
            )
        if self.dimensions is not None and not 64 <= self.dimensions <= 4096:
            raise RuntimeWarning(
                f"{self.dimensions} is not a valid dimension size"                 
            )
        if self.batch_size < 1:
            raise RuntimeWarning(
                f"{self.batch_size} is not a valid batch size"                 
            )
            
        self.api_key = key
        return self
    
    def _embed(self, text : list[str], task: str) -> list[list[float]]:
        payload_base: dict[str, Any] = {"model": self.model, "task": task}
        if self.dimensions:
            payload_base["dimensions"] = self.dimensions
        if self.late_chunking and task == "retrieval.passage":
            payload_base["late_chunking"] = True
        resp.raise_for_status()
        