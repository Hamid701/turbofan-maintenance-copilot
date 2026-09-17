#!/usr/bin/env bash
# Create the dashboard, or update the existing one with the same display name.
#
#   bash monitoring/apply-dashboard.sh
#
# Cloud Monitoring rejects an update that does not carry the dashboard's current
# etag, its version marker, so that a change cannot silently overwrite someone
# else's. The etag belongs to the deployed dashboard, not to the file, so it is
# fetched and added to a temporary copy here.
set -euo pipefail

file=monitoring/dashboard.json
title=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['displayName'])" "$file")
id=$(gcloud monitoring dashboards list --filter="displayName=\"$title\"" --format='value(name)' | head -1)

if [ -z "$id" ]; then
  gcloud monitoring dashboards create --config-from-file="$file" >/dev/null
  echo "created: $title"
  exit 0
fi

etag=$(gcloud monitoring dashboards describe "$id" --format='value(etag)')
tmp=$(mktemp)
python3 - "$file" "$etag" >"$tmp" <<'PY'
import json
import sys

dashboard = json.load(open(sys.argv[1], encoding="utf-8"))
dashboard["etag"] = sys.argv[2]
print(json.dumps(dashboard))
PY
gcloud monitoring dashboards update "$id" --config-from-file="$tmp" >/dev/null
rm -f "$tmp"
echo "updated: $title"
