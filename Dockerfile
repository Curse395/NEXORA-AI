# NEXORA — Streamlit deployment image
# Build:  docker build -t nexora .
# Run:    docker run -p 8501:8501 nexora
FROM python:3.10-slim

# System fonts so matplotlib can render Devanagari/Bengali/etc. in heatmaps
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-core \
        fonts-noto \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit server config (also enforced in .streamlit/config.toml)
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# M2M100 will download (~1.9 GB) on first translation, then cache here.
VOLUME ["/app/models/m2m100"]

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]