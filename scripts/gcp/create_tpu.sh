#!/bin/bash
# Creates (or manually re-creates) the Abalone AlphaZero TPU pod slice as a
# spot queued resource. You normally only run this once, by hand, for the
# first launch -- after that, scripts/gcp/watch_tpu.sh (running on the
# watcher VM, see create_watcher_vm.sh) issues an equivalent request
# automatically whenever the slice is preempted.
#
# Fill in the variables below for your project. Double-check
# ACCELERATOR_TYPE/RUNTIME_VERSION against your zone before running:
#   gcloud compute tpus tpu-vm versions list --zone=$ZONE
#   gcloud compute tpus accelerator-types list --zone=$ZONE
# TPU CLI flags do change between gcloud releases -- if this command errors,
# check `gcloud compute tpus queued-resources create --help` against what's
# below rather than assuming the script is wrong.
set -euo pipefail

PROJECT_ID="your-project-id"           # abalonemcts
ZONE="europe-west4-a"                  # closest v6e zone to France; us-east1-d is the other v6e option
ACCELERATOR_TYPE="v6e-16"              # e.g. v6e-16 = 16 chips = 4 hosts of 4 chips; start smaller for validation
RUNTIME_VERSION="v2-alpha-tpuv6e"      # verify with `versions list` above
NODE_ID="abalone-v6e"                  # stable TPU VM name -- keep this identical across recreations
GCS_BUCKET="your-bucket-name"

QR_ID="abalone-qr-$(date +%s)"         # queued-resource request IDs must be unique each time

gcloud compute tpus queued-resources create "$QR_ID" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --node-id="$NODE_ID" \
  --accelerator-type="$ACCELERATOR_TYPE" \
  --runtime-version="$RUNTIME_VERSION" \
  --spot \
  --metadata-from-file=startup-script=scripts/tpu_startup.sh \
  --metadata="abalone-gcs-bucket=$GCS_BUCKET,abalone-iterations=200,abalone-games-per-iter=64,abalone-save-frequency=1"

echo "Requested $QR_ID for node $NODE_ID. Check progress with:"
echo "  gcloud compute tpus queued-resources describe $QR_ID --project=$PROJECT_ID --zone=$ZONE"
