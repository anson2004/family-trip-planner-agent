#!/usr/bin/env bash
# Copy the API keys from your .env into Google Secret Manager, so they never pass through OpenTofu.
# Run it again after changing a key; it only adds a new version when the value changed.
#
#   infra/set-secrets.sh <project-id> [path/to/.env]
set -euo pipefail

project="${1:?usage: infra/set-secrets.sh <project-id> [path/to/.env]}"
env_file="${2:-.env}"
[[ -f "$env_file" ]] || { echo "$env_file not found" >&2; exit 1; }

gcloud services enable secretmanager.googleapis.com --project "$project"

for name in GROQ_API_KEY SERPAPI_API_KEY TAVILY_API_KEY RESEND_API_KEY; do
  # Last NAME=value line; strip surrounding quotes.
  value="$(grep -E "^${name}=" "$env_file" | tail -n 1 | cut -d= -f2- || true)"
  value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
  id="$(echo "$name" | tr 'A-Z_' 'a-z-')"
  if [[ -z "$value" ]]; then
    echo "skip   $name (empty in $env_file; remove it from 'secrets' in terraform.tfvars)"
    continue
  fi
  if ! gcloud secrets describe "$id" --project "$project" >/dev/null 2>&1; then
    gcloud secrets create "$id" --project "$project" --replication-policy=automatic --quiet
  fi
  current="$(gcloud secrets versions access latest --secret "$id" --project "$project" 2>/dev/null || true)"
  if [[ "$current" == "$value" ]]; then
    echo "same   $id"
    continue
  fi
  printf '%s' "$value" | gcloud secrets versions add "$id" --project "$project" --data-file=- >/dev/null
  # Only the newest version stays active (old ones count towards the free 6 active versions).
  for old in $(gcloud secrets versions list "$id" --project "$project" --filter="state=ENABLED" \
      --sort-by="~createTime" --format="value(name)" | tail -n +2); do
    gcloud secrets versions destroy "$old" --secret "$id" --project "$project" --quiet >/dev/null
  done
  echo "stored $id"
done
