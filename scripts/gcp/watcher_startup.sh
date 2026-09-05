#!/bin/bash
# Startup-script for the always-on e2-micro watcher VM -- self-contained on
# purpose so it can be pasted directly into the Console's "Automation ->
# Startup script" field when creating the VM, with no gsutil/gcloud needed
# from your own machine. It embeds tpu_startup.sh's content itself (see the
# heredoc below), installs the Cloud SDK, and schedules a check every 5
# minutes that recreates the Abalone TPU node whenever it has disappeared
# (spot preemption deletes TPU VMs outright -- they can't be "restarted").
#
# Configure via instance metadata when creating this VM:
#   abalone-project-id        *required*  e.g. abalonemcts
#   abalone-tpu-zone          *required*  e.g. europe-west4-a
#   abalone-tpu-accelerator   *required*  e.g. v6e-8
#   abalone-tpu-runtime       *required*  e.g. v2-alpha-tpuv6e
#   abalone-tpu-node-id       *required*  e.g. abalone-v6e-validation
#   abalone-gcs-bucket        *required*  e.g. abalonemcts-training
#   abalone-iterations        (default: 200, passed through to the TPU's own metadata)
#   abalone-games-per-iter    (default: 64)
#   abalone-save-frequency    (default: 1)
#
# IMPORTANT: if you change how tpu_startup.sh works in the repo, you must
# update the embedded copy below too and re-paste this file into the
# watcher VM's metadata (Edit instance -> Automation -> Startup script) --
# it does NOT read tpu_startup.sh from the repo at runtime.
set -euo pipefail

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
    2>/dev/null || echo "$2"
}

log() { logger -t abalone-watcher-setup "$*"; echo "$(date -u +%FT%TZ) $*"; }

mkdir -p /opt/abalone-watcher

# --- Embedded copy of scripts/tpu_startup.sh, passed to the recreated TPU as
# its own startup-script metadata.
cat > /opt/abalone-watcher/tpu_startup.sh <<'TPU_STARTUP_EOF'
#!/bin/bash
set -euo pipefail

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
    2>/dev/null || echo "$2"
}

REPO_URL="$(meta abalone-repo-url https://github.com/Mr-Corentin/abalone-alphazero.git)"
REPO_DIR="/root/abalone-alphazero"

if ! systemctl is-active --quiet google-cloud-ops-agent 2>/dev/null; then
  curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash add-google-cloud-ops-agent-repo.sh --also-install
  rm -f add-google-cloud-ops-agent-repo.sh
fi

if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
chmod +x scripts/run_training.sh

pip install -r requirements.txt
pip install "jax[tpu]==0.4.30" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

cat > /etc/systemd/system/abalone-training.service <<EOF
[Unit]
Description=Abalone AlphaZero training
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
ExecStart=$REPO_DIR/scripts/run_training.sh
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now abalone-training.service
TPU_STARTUP_EOF

# --- The actual watcher logic, run every 5 minutes by cron below.
cat > /opt/abalone-watcher/watch_tpu.sh <<'WATCH_EOF'
#!/bin/bash
set -euo pipefail

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
    2>/dev/null || echo "$2"
}

PROJECT_ID="$(meta abalone-project-id)"
ZONE="$(meta abalone-tpu-zone)"
ACCELERATOR_TYPE="$(meta abalone-tpu-accelerator)"
RUNTIME_VERSION="$(meta abalone-tpu-runtime)"
NODE_ID="$(meta abalone-tpu-node-id)"
GCS_BUCKET="$(meta abalone-gcs-bucket)"
ITERATIONS="$(meta abalone-iterations 200)"
GAMES_PER_ITER="$(meta abalone-games-per-iter 64)"
SAVE_FREQUENCY="$(meta abalone-save-frequency 1)"
# "true" (default) requests spot/preemptible capacity; "false" requests
# on-demand -- e.g. for the 32-chip v4 on-demand quota, which isn't subject
# to the preemption/stockout cycle spot capacity is.
SPOT="$(meta abalone-tpu-spot true)"
# Forwarded to the TPU's own metadata below -- these live on the WATCHER's
# metadata but run_training.sh reads them from the TPU it ends up running on,
# so they have to be passed through explicitly or they're silently empty on
# the recreated node (this is exactly how --num-simulations got dropped and
# training silently ran at the config default instead).
EXTRA_ARGS="$(meta abalone-extra-args "")"
VERTEX_TENSORBOARD_ID="$(meta abalone-vertex-tensorboard-id "")"
GCP_PROJECT_META="$(meta abalone-gcp-project "")"
GCP_LOCATION_META="$(meta abalone-gcp-location "")"

log() { logger -t abalone-watcher "$*"; echo "$(date -u +%FT%TZ) $*"; }

STATE="$(gcloud compute tpus tpu-vm describe "$NODE_ID" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --format='value(state)' 2>/dev/null || echo "MISSING")"

if [ "$STATE" = "READY" ]; then
  log "TPU node $NODE_ID is READY, nothing to do"
  exit 0
fi

IN_FLIGHT="$(gcloud compute tpus queued-resources list \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --filter="name~^abalone-qr AND (state.state=WAITING_FOR_RESOURCES OR state.state=PROVISIONING OR state.state=ACCEPTED)" \
  --format='value(name)' 2>/dev/null || true)"

if [ -n "$IN_FLIGHT" ]; then
  log "Queued resource $IN_FLIGHT already in flight for $NODE_ID, nothing to do"
  exit 0
fi

log "TPU node $NODE_ID state=$STATE, requesting a new slice (spot=$SPOT)"
QR_ID="abalone-qr-$(date +%s)"

METADATA="abalone-gcs-bucket=$GCS_BUCKET,abalone-iterations=$ITERATIONS,abalone-games-per-iter=$GAMES_PER_ITER,abalone-save-frequency=$SAVE_FREQUENCY"
[ -n "$EXTRA_ARGS" ] && METADATA="$METADATA,abalone-extra-args=$EXTRA_ARGS"
[ -n "$VERTEX_TENSORBOARD_ID" ] && METADATA="$METADATA,abalone-vertex-tensorboard-id=$VERTEX_TENSORBOARD_ID"
[ -n "$GCP_PROJECT_META" ] && METADATA="$METADATA,abalone-gcp-project=$GCP_PROJECT_META"
[ -n "$GCP_LOCATION_META" ] && METADATA="$METADATA,abalone-gcp-location=$GCP_LOCATION_META"

CREATE_ARGS=(
  --project="$PROJECT_ID"
  --zone="$ZONE"
  --node-id="$NODE_ID"
  --accelerator-type="$ACCELERATOR_TYPE"
  --runtime-version="$RUNTIME_VERSION"
  --metadata-from-file=startup-script=/opt/abalone-watcher/tpu_startup.sh
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
WATCH_EOF
chmod +x /opt/abalone-watcher/watch_tpu.sh

# --- gcloud isn't preinstalled on the plain Debian image -- install it once.
if ! command -v gcloud >/dev/null 2>&1; then
  log "Installing Google Cloud SDK"
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    > /etc/apt/sources.list.d/google-cloud-sdk.list
  curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  apt-get update -y
  apt-get install -y google-cloud-cli
fi

cat > /etc/cron.d/abalone-watcher <<CRON
*/5 * * * * root /opt/abalone-watcher/watch_tpu.sh >> /var/log/abalone-watcher.log 2>&1
CRON

if ! systemctl is-active --quiet google-cloud-ops-agent 2>/dev/null; then
  curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash add-google-cloud-ops-agent-repo.sh --also-install
  rm -f add-google-cloud-ops-agent-repo.sh
fi

log "Watcher setup complete, running its first check now"
/opt/abalone-watcher/watch_tpu.sh || true
