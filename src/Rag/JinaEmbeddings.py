from __future__ import annotations

import time
from typing import Any
import requests

from langchain_core.embeddings import Embeddings

from pydantic import BaseModel, model_validator, PrivateAttr

from src.utils.Settings import settings

API_URL = settings.JINA_API_URL
DEFAULT_MODEL = settings.JINA_MODEL 
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
        key = self.api_key or settings.JINA_API_KEY
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
                    
    def _post(self, payload: dict[str, any]) -> list[list[float]]:
        last_error: Exception | None = None
        
        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.post(
                    API_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=TIMEOUT,
                )
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(2**attempt)
                continue
                
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = RuntimeError(f"Jina API {resp.status_code}: {resp.text[:200]}")
                time.sleep(2**attempt)
                continue
            if resp.status_code == 401:
                raise RuntimeError(f"JINA_API_KEY was rejected status code : {resp.status_cod}")
            if not resp.ok:
                raise RuntimeError(f"Jina API {resp.status_code}: {resp.text[:300]}")

            data = resp.json().get("data")
            if not data:
                raise RuntimeError(f"Jina has no response 'data': {resp.text[:200]}")
            
            return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
        
        raise RuntimeError(f"Jina API failed after : {MAX_RETRIES} tries: {last_error}")
    
    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        payload_base: dict[str, Any] = {"model": self.model, "task": task}
        if self.dimensions:
            payload_base["dimensions"] = self.dimensions
        if self.late_chunking and task == "retrieval.passage":
            payload_base["late_chunking"] = True

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._post({**payload_base, "input": batch}))

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"jumlah embedding ({len(vectors)}) != jumlah input ({len(texts)})"
            )
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk dokumen — pakai task retrieval.passage."""
        cleaned = [t if t.strip() else " " for t in texts]
        if not cleaned:
            return []
        return self._embed(cleaned, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        """Embed query user — pakai task retrieval.query."""
        if not text.strip():
            raise ValueError("query kosong")
        return self._embed([text], task="retrieval.query")[0]