#!/bin/bash
# TPU VM startup-script: runs automatically on every boot, including after a
# spot preemption + recreation. Installs dependencies, wires up Cloud Logging,
# and (re)starts training as a systemd service so both a full VM recreation
# AND a plain process crash auto-resume without anyone SSHing in.
#
# Runs identically and independently on every worker of a multi-host slice
# (each VM gets its own copy of this same metadata script at boot) -- by the
# time any worker restarts, GCS checkpoint state is stable (no live writer),
# so each worker resolving "the latest checkpoint" on its own is safe and
# every worker ends up making the same choice.
#
# Deploy by passing this file's contents as the "startup-script" metadata key
# when creating the TPU (see gcp/create_tpu.sh), together with these
# attribute keys (all except abalone-gcs-bucket have defaults, see
# run_training.sh):
#   abalone-repo-url        (default: this repo's GitHub URL)
#   abalone-gcs-bucket       *required*
#   abalone-iterations       (default: 200)
#   abalone-games-per-iter   (default: 64)
#   abalone-save-frequency   (default: 1 -- see the comment below on why)
#   abalone-extra-args       (default: "", passed through verbatim to main.py)
set -euo pipefail

meta() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
    2>/dev/null || echo "$2"
}

REPO_URL="$(meta abalone-repo-url https://github.com/Mr-Corentin/abalone-alphazero.git)"
REPO_DIR="/root/abalone-alphazero"

# --- Observability: forwards this VM's systemd/journal logs to Cloud Logging
# and adds extra host-level metrics to Cloud Monitoring. TPU chip metrics
# (TensorCore/HBM utilization) already flow to Monitoring on their own
# without this agent -- it's for the training process's own log lines.
if ! systemctl is-active --quiet google-cloud-ops-agent 2>/dev/null; then
  curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash add-google-cloud-ops-agent-repo.sh --also-install
  rm -f add-google-cloud-ops-agent-repo.sh
fi

# --- Code
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
chmod +x scripts/run_training.sh

# --- Deps. jax[tpu] MUST be installed after requirements.txt: requirements.txt
# pins a generic (CPU) jaxlib, and jax[tpu] needs to win so jax.devices()
# actually reports TPU devices.
#
# Pinned to the SAME version as requirements.txt's jax/jaxlib entries (not -U
# / latest): flax==0.8.5 reaches into a JAX trace-stack internal
# (flax/core/tracers.py's trace_level) that a much newer JAX no longer
# exposes the same way, which surfaces as
# "AttributeError: 'EvalTrace' object has no attribute 'level'" the moment
# the network is initialized. Pinning keeps jax/jaxlib matched to the
# flax/chex/optax versions they were presumably tested against.
pip install -r requirements.txt
pip install "jax[tpu]==0.4.30" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# --- Run training as a systemd service. Restart=on-failure gives us
# process-level auto-restart on a plain crash/OOM, on top of the
# queued-resources recreation the watcher VM (see gcp/watch_tpu.sh) handles
# for full spot preemptions. Checkpoint resolution happens fresh inside
# run_training.sh on every (re)start, not here, precisely so this also covers
# a crash-restart hours into a run picking up the latest checkpoint rather
# than whatever existed at boot.
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
