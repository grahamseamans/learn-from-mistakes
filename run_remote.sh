#!/bin/bash

# Remote machine details
REMOTE_HOST="root@ssha.jarvislabs.ai"
REMOTE_PORT="11714"
REMOTE_DIR="/root/learn-from-mistakes"

# Sync code to remote machine (excluding unnecessary directories)
echo "Syncing code to remote machine..."
rsync -avz -e "ssh -p $REMOTE_PORT" \
    --exclude 'data/' \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    . "$REMOTE_HOST:$REMOTE_DIR"

# Run the specified script on remote machine
echo "Running on remote machine..."
ssh -p $REMOTE_PORT $REMOTE_HOST "cd $REMOTE_DIR && python3 $1" 