#!/bin/bash
# Clean Docker build + start for MVP-Echo Scribe
#
# Usage:
#   ./start.sh          # NeMo (default)
#   ./start.sh nemo     # NeMo explicitly
#   ./start.sh sherpa   # Sherpa-ONNX
#
# Stops the running container, removes old image, rebuilds, and starts fresh.
# Model volume (scribe-models) is preserved — no re-download between builds.

set -e

ENGINE="${1:-nemo}"

case "$ENGINE" in
    nemo)
        export DOCKERFILE=Dockerfile
        export ENGINE=nemo
        echo "=== MVP-Echo Scribe: NeMo stack ==="
        ;;
    sherpa)
        export DOCKERFILE=Dockerfile.sherpa
        export ENGINE=sherpa
        echo "=== MVP-Echo Scribe: Sherpa-ONNX stack ==="
        ;;
    *)
        echo "Unknown engine: $ENGINE"
        echo "Usage: $0 [nemo|sherpa]"
        exit 1
        ;;
esac

echo "  Dockerfile: $DOCKERFILE"
echo "  Engine:     $ENGINE"
echo ""

# Stop and remove container
echo "[1/3] Stopping container..."
docker compose down 2>/dev/null || true

# Remove old image to force clean build
echo "[2/3] Removing old image..."
docker rmi mvp-echo-scribe-mvp-scribe 2>/dev/null || true

# Build and start
echo "[3/3] Building and starting..."
docker compose up -d --build

echo ""
echo "=== Started ==="
echo "  Health:  curl http://localhost:20301/health"
echo "  Logs:    docker compose logs -f mvp-scribe"
echo "  Shell:   docker exec -it mvp-scribe bash"
