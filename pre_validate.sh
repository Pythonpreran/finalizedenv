#!/usr/bin/env bash
# OpenEnv Round-1 Pre-Validation Script
# Run this before submitting to verify compliance.
set -e

API_URL="${API_BASE_URL:-http://localhost:7860}"

echo "=== OpenEnv Pre-Validation ==="

# 1. Ping /reset to verify server is alive
echo "[1/3] Pinging ${API_URL}/reset ..."
curl -sf -X POST "${API_URL}/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}' \
  > /dev/null
echo "  ✓ Server responded to /reset"

# 2. Build Docker image
echo "[2/3] Building Docker image ..."
docker build -t icu-drug-titration .
echo "  ✓ Docker build succeeded"

# 3. Run openenv validate (if openenv CLI is installed)
echo "[3/3] Running openenv validate ..."
if command -v openenv &> /dev/null; then
  openenv validate
  echo "  ✓ openenv validate passed"
else
  echo "  ⚠ openenv CLI not found — skipping validate"
  echo "    Install with: pip install openenv"
fi

echo ""
echo "=== All pre-validation checks passed ==="
