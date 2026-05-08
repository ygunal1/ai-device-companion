#!/bin/bash
# Run this once on the VM as user 'deployer' to set everything up.
# Usage: bash setup.sh

set -e

APP_DIR="$HOME/whisplay-server"
SERVICE_NAME="whisplay-logs"

echo "=== Installing system dependencies ==="
sudo apt-get update -q
sudo apt-get install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx

echo "=== Creating app directory ==="
mkdir -p "$APP_DIR"
cp main.py requirements.txt "$APP_DIR/"

echo "=== Setting up Python virtualenv ==="
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "=== Generating API key ==="
API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

echo "=== Writing .env ==="
cat > "$APP_DIR/.env" <<EOF
API_KEY=$API_KEY
DB_PATH=$APP_DIR/logs.db
EOF
chmod 600 "$APP_DIR/.env"

echo ""
echo "  *** Save this API key — you'll put it in the Pi's .env as TRANSCRIPT_API_KEY ***"
echo "  API_KEY=$API_KEY"
echo ""

echo "=== Installing systemd service ==="
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Whisplay Log Collector
After=network.target

[Service]
User=deployer
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "=== Configuring nginx ==="
sudo cp nginx.conf /etc/nginx/sites-available/whisplay
sudo ln -sf /etc/nginx/sites-available/whisplay /etc/nginx/sites-enabled/whisplay
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

echo "=== Obtaining TLS certificate via Let's Encrypt ==="
sudo certbot --nginx -d aibuddy.ifi.uzh.ch --non-interactive --agree-tos -m your-email@uzh.ch

echo ""
echo "=== Done! ==="
echo "  POST logs to : https://aibuddy.ifi.uzh.ch/logs"
echo "  Download CSV : https://aibuddy.ifi.uzh.ch/export.csv  (needs X-API-Key header)"
echo "  Health check : https://aibuddy.ifi.uzh.ch/health"
