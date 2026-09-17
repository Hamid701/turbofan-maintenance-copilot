#!/usr/bin/env bash
# Create the alert policies in monitoring/alerts, sending their notifications to
# one channel.
#
#   bash monitoring/apply-alerts.sh projects/PROJECT_ID/notificationChannels/CHANNEL_ID
#
# Find the channel's resource name with:
#   gcloud alpha monitoring channels list --format='value(name,displayName)'
#
# Each file carries the literal CHANNEL in place of that name; this script
# substitutes it into a temporary copy and never edits the files themselves.
# A policy whose display name already exists is skipped, so re-running is safe.
set -euo pipefail

channel=${1:?Pass the notification channel resource name}

for file in monitoring/alerts/*.json; do
  title=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['displayName'])" "$file")
  if [ -n "$(gcloud monitoring policies list --filter="displayName=\"$title\"" --format='value(name)')" ]; then
    echo "exists, skipped: $title"
    continue
  fi
  tmp=$(mktemp)
  python3 - "$file" "$channel" >"$tmp" <<'PY'
import json
import sys

policy = json.load(open(sys.argv[1], encoding="utf-8"))
policy["notificationChannels"] = [sys.argv[2]]
print(json.dumps(policy))
PY
  # One rejected file must not stop the rest: Cloud Monitoring validates each
  # condition on creation, and a bad one should be reported, not block the others.
  if gcloud monitoring policies create --policy-from-file="$tmp" >/dev/null; then
    echo "created: $title"
  else
    echo "FAILED:  $title (see the error above)"
  fi
  rm -f "$tmp"
done
