#!/usr/bin/env bash
# Create or update every log-based metric in monitoring/metrics.
#
# Run from the repository root, in Cloud Shell or anywhere gcloud is signed in:
#   bash monitoring/apply-metrics.sh
#
# A metric counts only log entries written after it exists, so send a few
# questions through the service afterwards before expecting data.
set -euo pipefail

for file in monitoring/metrics/*.yaml; do
  name=$(awk '/^name:/ {print $2; exit}' "$file")
  if gcloud logging metrics describe "$name" >/dev/null 2>&1; then
    gcloud logging metrics update "$name" --config-from-file="$file" >/dev/null
    echo "updated $name"
  else
    gcloud logging metrics create "$name" --config-from-file="$file" >/dev/null
    echo "created $name"
  fi
done
