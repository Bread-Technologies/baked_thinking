#!/bin/bash

set -e

REMOTE_HOST="bread69"
REMOTE_PATH="/home/neel/Bread/oracle_cleaner"   # adjust if needed
BUNDLE="oracle_bundle.tar.gz"

echo "==> Packaging files..."
tar -czf $BUNDLE \
  scripts/oracle_cleaner.py \
  results/target_*.jsonl \
  external_benchmarks/.env \
  requirements.txt

echo "==> Copying bundle to $REMOTE_HOST:$REMOTE_PATH ..."
ssh $REMOTE_HOST "mkdir -p $REMOTE_PATH"
scp $BUNDLE $REMOTE_HOST:$REMOTE_PATH/

echo "==> Running remote setup..."

ssh $REMOTE_HOST << EOF
cd $REMOTE_PATH

echo "==> Extracting..."
tar -xzf $BUNDLE

echo "==> Installing dependencies..."
pip install -r requirements.txt || pip install anthropic openai google-generativeai python-dotenv tqdm

echo "==> Starting tmux session 'oracle'..."
tmux kill-session -t oracle 2>/dev/null || true
tmux new -d -s oracle "python scripts/oracle_cleaner.py"

echo "==> Done. Attach with: tmux attach -t oracle"
EOF

echo "==> Deployment complete."
