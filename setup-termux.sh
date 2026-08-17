#!/data/data/com.termux/files/usr/bin/bash
# Termux one-shot setup for Lucy (phone-only, on-device).
# Run from inside the Lucy-latest folder:  bash setup-termux.sh
set -e

cd "$(dirname "$0")"

echo "=== Updating Termux packages ==="
pkg update -y
pkg install python git -y

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

if [ ! -f training/checkpoints/latest.json ]; then
  echo "=== Training on-device model (a few minutes, pure Python) ==="
  python3 -m training.train
else
  echo "=== Checkpoint already present, skipping training ==="
fi

echo "=== Starting Lucy chat ==="
echo "Type: good morning | run check system health | state | trust | goodnight | exit"
python3 lucy_cli.py --config lucy.yaml
