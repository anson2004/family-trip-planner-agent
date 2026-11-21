#!/usr/bin/env bash
# Deploy the newest image with OpenTofu. The deploy job in pipeline.yml runs this; you can run it locally too.
#
#   PROJECT_ID=<project-id> infra/deploy.sh          # apply (asks first, unless in GitHub Actions)
#   PROJECT_ID=<project-id> infra/deploy.sh plan     # only show what would change
#
# terraform.tfvars (settings: trip, email...) is shared through the state bucket:
#   - locally, your infra/terraform.tfvars is uploaded first, so your edits become the shared config;
#   - in GitHub Actions (CI=true) it is downloaded from the bucket.
# IMAGE=<image> overrides the image (default: the newest one pushed by the build job).
set -euo pipefail
cd "$(dirname "$0")"

project="${PROJECT_ID:?set PROJECT_ID}"
region="${REGION:-europe-north1}"
bucket="${project}-tfstate"
config="gs://${bucket}/config/terraform.tfvars"

if [[ "${CI:-}" == "true" ]]; then
  gcloud storage cp "$config" terraform.tfvars --quiet
else
  gcloud storage cp terraform.tfvars "$config" --quiet
fi

if [[ -z "${IMAGE:-}" ]]; then
  repo="${region}-docker.pkg.dev/${project}/travel-plan/app"
  digest="$(gcloud artifacts docker images describe "${repo}:latest" --format='value(image_summary.digest)')"
  IMAGE="${repo}@${digest}"
fi
echo "image: $IMAGE"

# -migrate-state moves an existing local state into the bucket the first time; later it does nothing.
tofu init -input=false -migrate-state -force-copy -backend-config="bucket=${bucket}"

if [[ "${1:-apply}" == "plan" ]]; then
  tofu plan -input=false -var "image=${IMAGE}"
elif [[ "${CI:-}" == "true" ]]; then
  tofu apply -input=false -auto-approve -var "image=${IMAGE}"
else
  tofu apply -var "image=${IMAGE}"
fi
