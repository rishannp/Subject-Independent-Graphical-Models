#!/bin/bash

SESSION_NAME="train_hopefulnet"
SCRIPT_NAME="/home/uceerjp/StiegerModel/1DCNN/HopefulNet.py"
LOG_FILE="HopefullNet_train_output.log"
RETRY_INTERVAL=60
PYTHON_EXEC="python"

# --- Pick GPU interactively ---
echo "[INFO] Available GPUs:"
nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu \
  --format=csv,noheader,nounits | \
  awk '{printf "GPU %s: %s MiB free / %s MiB total | Utilization: %s%%\n", $1, $2, $3, $4}'

echo ""
read -p "[INPUT] Enter GPU ID to use (0,1,2,...): " SELECTED_GPU
export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"

# --- Launch training in tmux ---
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[INFO] Tmux session '$SESSION_NAME' already exists."
else
  echo "[INFO] Launching tmux training session on GPU $SELECTED_GPU..."
  tmux new-session -d -s "$SESSION_NAME" \
    "bash -c 'while true; do \
      echo \"[START] Training at \$(date) on GPU $SELECTED_GPU\" >> $LOG_FILE; \
      $PYTHON_EXEC -u $SCRIPT_NAME 2>&1 | tee -a $LOG_FILE; \
      echo \"[RESTART] Script crashed or ended at \$(date), retrying in ${RETRY_INTERVAL}s...\" >> $LOG_FILE; \
      sleep $RETRY_INTERVAL; \
    done'"
  echo "[SUCCESS] Training running in tmux session '$SESSION_NAME'."
  echo "Use 'tmux attach -t $SESSION_NAME' to watch it live."
fi
