#!/bin/bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG="configs/train_semisup_config.yaml"
NUM_GPUS=${NUM_GPUS:-1}
BACKEND=${BACKEND:-nccl}
LOG_DIR="logs/run_logs"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "Starting training at $(date)"
echo "Config:   $CONFIG"
echo "GPUs:     $NUM_GPUS"
echo "Backend:  $BACKEND"
echo "Log file: $LOG_FILE"
echo "──────────────────────────────────────────────────"

# ── Launch ────────────────────────────────────────────────────────────────────
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    train_semisup.py \
    --config "$CONFIG" \
    --backend "$BACKEND" \
    2>&1 | tee "$LOG_FILE"

echo "──────────────────────────────────────────────────"
echo "Training complete at $(date)"
echo "Log saved to $LOG_FILE"
