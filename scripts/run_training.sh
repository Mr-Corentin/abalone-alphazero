#!/bin/bash
# Wrapper invoked by the abalone-training systemd unit (see tpu_startup.sh).
#
# This is a separate script -- not inlined into the systemd unit -- because it
# has to re-resolve the latest checkpoint EVERY time it runs, not just once at
# VM boot. systemd's Restart=on-failure re-executes the unit's ExecStart
# verbatim; if the checkpoint path were baked into the unit file at boot time,
# a crash-restart hours later would resume from that stale boot-time
# checkpoint instead of whatever was saved most recently.
#
# Also fixes a related trap: main.py, given only --gcs-bucket without an
# explicit --checkpoint-path, generates a NEW timestamped checkpoint prefix on
# every launch (see get_merged_config in main.py) -- so a restart would use a
# different prefix than the previous run and find_latest_checkpoint.py would
# never see any of the earlier checkpoints. --checkpoint-path is set
# explicitly here to keep it stable across every restart.
set -euo pipefail
cd "$(dirname "$0")/.."

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
    2>/dev/null || echo "$2"
}

GCS_BUCKET="$(meta abalone-gcs-bucket "")"
ITERATIONS="$(meta abalone-iterations 200)"
GAMES_PER_ITER="$(meta abalone-games-per-iter 64)"
SAVE_FREQUENCY="$(meta abalone-save-frequency 1)"
EXTRA_ARGS="$(meta abalone-extra-args "")"

if [ -z "$GCS_BUCKET" ]; then
  echo "abalone-gcs-bucket metadata key is required, aborting" >&2
  exit 1
fi

CHECKPOINT_PATH="gs://${GCS_BUCKET}/checkpoints/model"

RESUME_ARGS=()
if LATEST=$(python3 scripts/find_latest_checkpoint.py "$CHECKPOINT_PATH" 2>/dev/null); then
  echo "Resuming from checkpoint: $LATEST"
  RESUME_ARGS=(--checkpoint "$LATEST")
else
  echo "No existing checkpoint found under $CHECKPOINT_PATH, starting fresh"
fi

exec python3 main.py --mode train \
  --gcs-bucket "$GCS_BUCKET" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --iterations "$ITERATIONS" \
  --games-per-iter "$GAMES_PER_ITER" \
  --save-frequency "$SAVE_FREQUENCY" \
  "${RESUME_ARGS[@]}" $EXTRA_ARGS
