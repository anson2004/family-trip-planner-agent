#!/usr/bin/env bash
# One-time: create the bucket that holds the OpenTofu state and terraform.tfvars, shared by your Mac
# and GitHub Actions. OpenTofu can't create the bucket it keeps its own state in, so this uses gcloud.
#
#   infra/bootstrap-state.sh <project-id> [region]
set -euo pipefail

project="${1:?usage: infra/bootstrap-state.sh <project-id> [region]}"
region="${2:-europe-north1}"
bucket="gs://${project}-tfstate"

if gcloud storage buckets describe "$bucket" --project "$project" >/dev/null 2>&1; then
  echo "exists  $bucket"
else
  gcloud storage buckets create "$bucket" --project "$project" --location "$region" \
    --uniform-bucket-level-access --public-access-prevention
  echo "created $bucket"
fi
# Every change to the state or config keeps the old copy, so a bad apply can be undone.
gcloud storage buckets update "$bucket" --versioning --quiet >/dev/null
echo "versioning on"
