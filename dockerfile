# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Nonaktifkan .pyc dan buffering supaya log Streamlit langsung keluar
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
# Deps dulu, biar layer ini ke-cache selama requirements.txt tidak berubah
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY data/ ./data/

# Jalan sebagai non-root; folder data dan index harus bisa ditulis user ini
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/.chroma_db \
    && chown -R app:app /app
USER app

ENV SEMSEARCH_DB=/app/.chroma_db \
    SEMSEARCH_UPLOADS=/app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
