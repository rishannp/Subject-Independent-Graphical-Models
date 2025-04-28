#!/bin/bash

SESSION_NAME="train_lmda"
SCRIPT_NAME="/home/uceerjp/StiegerModel/LMDA/LMDA.py"
LOG_FILE="LMDA_train_output.log"
RETRY_INTERVAL=60  # If script crashes
GPU_CHECK_INTERVAL=1  # How often to recheck GPUs
MEMORY_THRESHOLD=40000  # Minimum 40 GB free
PYTHON_EXEC="python"

# --- Function to check for free GPUs ---
find_free_gpu() {
  while true; do
    echo "[INFO] Checking GPUs for available memory..."
    AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

    while read -r line; do
      GPU_ID=$(echo $line | awk '{print $1}')
      FREE_MEM=$(echo $line | awk '{print $2}')
      if [ "$FREE_MEM" -ge "$MEMORY_THRESHOLD" ]; then
        echo "[SUCCESS] GPU $GPU_ID has $FREE_MEM MiB free, selecting it."
        echo $GPU_ID
        return
      fi
    done <<< "$AVAILABLE_GPUS"

    echo "[WAIT] No suitable GPU found, retrying in ${GPU_CHECK_INTERVAL}s..."
    sleep $GPU_CHECK_INTERVAL
  done
}

# --- Main launcher ---
echo "[INFO] Looking for free GPU with at least ${MEMORY_THRESHOLD} MiB free..."

SELECTED_GPU=$(find_free_gpu)
export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"

echo "[INFO] Selected GPU $SELECTED_GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

# --- Launch tmux session ---
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
