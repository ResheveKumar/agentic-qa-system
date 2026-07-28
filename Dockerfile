# Minimal Dockerfile for the Agentic Codebase & Knowledge QA System
# Builds and runs the Streamlit app (app.py) as the container's entry point.

FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (better layer caching on rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY mcp_server.py rag_engine.py agent_graph.py app.py ./

# LanceDB vector store lives here at runtime; mount a volume to persist it
RUN mkdir -p /app/lancedb_store
VOLUME ["/app/lancedb_store"]

EXPOSE 8501

# GROQ_API_KEY must be supplied at runtime, e.g.:
#   docker run --env-file .env -p 8501:8501 agentic-qa-system
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py"]
