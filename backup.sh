#!/bin/bash
# Dated backup script - excludes pycache, temp files, and large runtime data dirs

BACKUP_DIR="$HOME/backups"
DATE=$(date +%Y-%m-%d_%H%M)
DEST="${BACKUP_DIR}/${DATE}_backup"

mkdir -p "$DEST"

rsync -a \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.lgd*' \
    --exclude='.pytest_cache' \
    --exclude='data_collection/hydraulic_data/' \
    --exclude='logs_hw/' \
    "$(pwd)/" "$DEST/"

echo "Backup created: $DEST"
