#!/bin/bash

# Remote machine details
REMOTE_HOST="root@ssha.jarvislabs.ai"
REMOTE_PORT="11214"
REMOTE_DIR="/root/learn-from-mistakes"
SSH_OPTS="-o StrictHostKeyChecking=no"

# Sync code to remote machine (excluding unnecessary directories)
echo "Syncing code to remote machine..."
rsync -avz -e "ssh $SSH_OPTS -p $REMOTE_PORT" \
    --exclude 'data/' \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    . "$REMOTE_HOST:$REMOTE_DIR"

# Run the specified script on remote machine
echo "Running on remote machine..."
ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST "cd $REMOTE_DIR && python3 $1" 