#!/bin/bash
# Build and run the FreightRates RAG API + chat UI on the shared Docker network
set -e

docker network create freight-net 2>/dev/null || true            # shared with the forecast containers
docker rm -f freight-rag freight-rag-ui 2>/dev/null || true       # remove old containers

# RAG API (Flask) - reachable from the UI as http://freight-rag:7870
docker build -t freight-rag ./rag_files
docker run -d --name freight-rag --network freight-net -p 7870:7870 freight-rag

# Chat UI (Streamlit)
docker build -t freight-rag-ui ./ui_files
docker run -d --name freight-rag-ui --network freight-net -p 8502:8502 \
  -e RAG_URL=http://freight-rag:7870 freight-rag-ui

docker ps
echo "RAG API -> port 7870 | Chat UI -> port 8502 (see the PORTS tab)."
echo "The API needs ~1 minute to load its models after start: docker logs -f freight-rag"
