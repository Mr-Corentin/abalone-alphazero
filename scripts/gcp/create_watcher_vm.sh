#!/bin/bash
# One-off: creates the always-on e2-micro watcher VM that recreates the
# Abalone TPU pod slice whenever a spot preemption destroys it.
#
# Cheap on purpose: e2-micro fits GCP's Always Free tier (720 hrs/month) in
# us-west1/us-central1/us-east1, and this VM only runs `gcloud`/`gsutil`
# calls every 5 minutes -- no framework, nothing to package or deploy beyond
# two shell scripts.
#
# Requires the default Compute Engine service account (or whichever you pass
# via --service-account) to have the "TPU Admin" role (roles/tpu.admin) and
# Storage access on $GCS_BUCKET -- grant once with:
#   gcloud projects add-iam-policy-binding $PROJECT_ID \
#     --member="serviceAccount:$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
#     --role="roles/tpu.admin"
set -euo pipefail

PROJECT_ID="your-project-id"
WATCHER_ZONE="us-east1-b"       # Always Free eligible; also close to the us-east1-d v6e quota if you use that zone
GCS_BUCKET="your-bucket-name"

# Run from the repo root so the relative paths below resolve.
gsutil cp scripts/tpu_startup.sh "gs://$GCS_BUCKET/deploy/tpu_startup.sh"
gsutil cp scripts/gcp/watch_tpu.sh "gs://$GCS_BUCKET/deploy/watch_tpu.sh"

WATCHER_STARTUP="$(mktemp)"
cat > "$WATCHER_STARTUP" <<EOF
#!/bin/bash
set -e
mkdir -p /opt/abalone-watcher
gsutil cp "gs://$GCS_BUCKET/deploy/watch_tpu.sh" /opt/abalone-watcher/watch_tpu.sh
chmod +x /opt/abalone-watcher/watch_tpu.sh

cat > /etc/cron.d/abalone-watcher <<CRON
*/5 * * * * root /opt/abalone-watcher/watch_tpu.sh >> /var/log/abalone-watcher.log 2>&1
CRON

if ! systemctl is-active --quiet google-cloud-ops-agent 2>/dev/null; then
  curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash add-google-cloud-ops-agent-repo.sh --also-install
fi
EOF

gcloud compute instances create abalone-watcher \
  --project="$PROJECT_ID" \
  --zone="$WATCHER_ZONE" \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script="$WATCHER_STARTUP"

rm -f "$WATCHER_STARTUP"
echo "Watcher VM created in $WATCHER_ZONE. It checks the TPU slice every 5 minutes."
echo "To update watch_tpu.sh config later: edit it, re-run the two gsutil cp lines above,"
echo "no need to recreate the VM (it re-fetches the script on every boot only though --"
echo "for an immediate update also run: gcloud compute ssh abalone-watcher --zone=$WATCHER_ZONE --command='sudo gsutil cp gs://$GCS_BUCKET/deploy/watch_tpu.sh /opt/abalone-watcher/watch_tpu.sh')"
