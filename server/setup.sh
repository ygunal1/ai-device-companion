#!/bin/bash
# Run once on the VM as user 'deployer' to set everything up.
# Usage: bash setup.sh

set -e

echo "=== Installing Docker ==="
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

echo "=== Generating API key ==="
API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

echo "=== Writing .env ==="
cat > .env <<EOF
API_KEY=$API_KEY
EOF
chmod 600 .env

echo ""
echo "  *** Save this API key — put it in each Pi's .env as TRANSCRIPT_API_KEY ***"
echo "  API_KEY=$API_KEY"
echo ""

echo "=== Starting stack ==="
# newgrp lets the docker group take effect without logging out
newgrp docker <<'NEWGRP'
docker compose up -d --build
NEWGRP

echo ""
echo "=== Done! ==="
echo "  POST logs : https://aibuddy.ifi.uzh.ch/logs          (X-API-Key header required)"
echo "  Export CSV: https://aibuddy.ifi.uzh.ch/export.csv    (X-API-Key header required)"
echo "  Health    : https://aibuddy.ifi.uzh.ch/health"
echo ""
echo "  To update every 2 weeks:"
echo "    docker compose pull && docker compose up -d"
