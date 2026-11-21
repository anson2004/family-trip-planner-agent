terraform {
  required_version = ">= 1.8"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0, < 8.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# The Google Cloud services this deployment uses. They stay on if you destroy, so nothing else breaks.
resource "google_project_service" "apis" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudscheduler.googleapis.com",
    "iam.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}
