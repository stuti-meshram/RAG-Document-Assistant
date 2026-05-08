#!/usr/bin/env bash
# scripts/deploy_ec2.sh
# ─────────────────────────────────────────────────────────
# Quick-deploy script for a single EC2 instance.
# Run this from the project root after SSHing in.
#
#   Usage:  bash scripts/deploy_ec2.sh
# ─────────────────────────────────────────────────────────
set -euo pipefail

IMAGE="rag-document-assistant"
CONTAINER="rag-assistant"
PORT=8501

echo "▶ Pulling latest image…"
docker pull "$IMAGE:latest" 2>/dev/null || echo "(local build — skipping pull)"

echo "▶ Stopping existing container (if any)…"
docker stop "$CONTAINER" 2>/dev/null || true
docker rm   "$CONTAINER" 2>/dev/null || true

echo "▶ Starting new container…"
docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  -p "$PORT:$PORT" \
  --env-file .env \
  -v rag_faiss:/app/data \
  "$IMAGE:latest"

echo "✅ Container '$CONTAINER' is running on port $PORT."
echo "   Visit: http://$(curl -s ifconfig.me):$PORT"
