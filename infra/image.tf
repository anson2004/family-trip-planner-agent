# Where the container images live. The Build workflow in GitHub Actions pushes them here;
# OpenTofu only deploys the image it is given (var.image).

resource "google_artifact_registry_repository" "app" {
  repository_id = "travel-plan"
  format        = "DOCKER"
  location      = var.region

  # Keep only the 2 newest images, so storage stays inside the free 0.5 GB.
  cleanup_policy_dry_run = false
  cleanup_policies {
    id     = "keep-newest"
    action = "KEEP"
    most_recent_versions {
      keep_count = 2
    }
  }
  cleanup_policies {
    id     = "delete-older"
    action = "DELETE"
    condition {
      older_than = "86400s"
    }
  }

  depends_on = [google_project_service.apis]
}
