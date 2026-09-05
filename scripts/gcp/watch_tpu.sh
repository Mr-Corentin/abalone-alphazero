#!/bin/bash
# Runs every ~5 minutes (via cron, see create_watcher_vm.sh) on the small
# always-on watcher VM. TPU spot VMs can't be "restarted" after preemption --
# Google deletes them outright -- so recreating the pod slice has to happen
# from somewhere that keeps running after the TPU itself is gone. That's the
# only job of this script. Once new TPU VMs exist, scripts/tpu_startup.sh
# (which reruns automatically on every boot of those VMs) handles actually
# resuming training from the latest checkpoint.
#
# Deployed to gs://$GCS_BUCKET/deploy/watch_tpu.sh by create_watcher_vm.sh --
# edit the variables below and re-run that deploy step to change config.
set -euo pipefail

PROJECT_ID="your-project-id"
ZONE="europe-west4-a"
ACCELERATOR_TYPE="v6e-16"
RUNTIME_VERSION="v2-alpha-tpuv6e"
NODE_ID="abalone-v6e"
GCS_BUCKET="your-bucket-name"
STARTUP_SCRIPT_GCS="gs://your-bucket-name/deploy/tpu_startup.sh"
SPOT=true  # false for an on-demand quota (e.g. the v4 on-demand allocation)
EXTRA_ARGS=""  # e.g. "--num-simulations 400" -- forwarded to the TPU's metadata below

log() { logger -t abalone-watcher "$*"; echo "$(date -u +%FT%TZ) $*"; }

STATE="$(gcloud compute tpus tpu-vm describe "$NODE_ID" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --format='value(state)' 2>/dev/null || echo "MISSING")"

if [ "$STATE" = "READY" ]; then
  log "TPU node $NODE_ID is READY, nothing to do"
  exit 0
fi

# Don't pile up a second request if one is already waiting for spot capacity.
IN_FLIGHT="$(gcloud compute tpus queued-resources list \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --filter="name~^abalone-qr AND (state.state=WAITING_FOR_RESOURCES OR state.state=PROVISIONING OR state.state=ACCEPTED)" \
  --format='value(name)' 2>/dev/null || true)"

if [ -n "$IN_FLIGHT" ]; then
  log "Queued resource $IN_FLIGHT already in flight for $NODE_ID, nothing to do"
  exit 0
fi

log "TPU node $NODE_ID state=$STATE, requesting a new slice"

LOCAL_STARTUP=/tmp/tpu_startup.sh
gsutil cp "$STARTUP_SCRIPT_GCS" "$LOCAL_STARTUP"

QR_ID="abalone-qr-$(date +%s)"

METADATA="abalone-gcs-bucket=$GCS_BUCKET,abalone-iterations=200,abalone-games-per-iter=64,abalone-save-frequency=1"
[ -n "$EXTRA_ARGS" ] && METADATA="$METADATA,abalone-extra-args=$EXTRA_ARGS"

CREATE_ARGS=(
  --project="$PROJECT_ID"
  --zone="$ZONE"
  --node-id="$NODE_ID"
  --accelerator-type="$ACCELERATOR_TYPE"
  --runtime-version="$RUNTIME_VERSION"
  --metadata-from-file=startup-script="$LOCAL_STARTUP"
  --metadata="$METADATA"
)
if [ "$SPOT" = "true" ]; then
  CREATE_ARGS+=(--spot)
fi

if gcloud compute tpus queued-resources create "$QR_ID" "${CREATE_ARGS[@]}"; then
  log "Requested $QR_ID for node $NODE_ID"
else
  log "Failed to request $QR_ID (capacity likely unavailable right now, will retry next tick)"
fi
