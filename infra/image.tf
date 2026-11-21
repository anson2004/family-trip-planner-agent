# The container image. Built by Cloud Build (no Docker needed on your machine) whenever the code changes.

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

locals {
  repo_root = abspath("${path.module}/..")
  # Everything that goes into the image. A change in any of these builds a new image.
  source_files = concat(
    [for f in fileset(local.repo_root, "src/**") : f if length(regexall("__pycache__", f)) == 0],
    ["Dockerfile", "pyproject.toml", "uv.lock", "README.md"],
  )
  source_hash = substr(sha1(join("", [for f in sort(local.source_files) : filesha1("${local.repo_root}/${f}")])), 0, 12)
  image       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.app.repository_id}/app:${local.source_hash}"
}

# Cloud Build runs builds as the project's default compute service account. New projects give it no
# rights, so grant what a build needs: read the uploaded source, write logs, push the image.
data "google_project" "this" {}

locals {
  build_account = "serviceAccount:${data.google_project.this.number}-compute@developer.gserviceaccount.com"
}

resource "google_project_iam_member" "build_reads_source" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = local.build_account
}

resource "google_project_iam_member" "build_writes_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = local.build_account
}

resource "google_artifact_registry_repository_iam_member" "build_pushes_image" {
  repository = google_artifact_registry_repository.app.name
  location   = var.region
  role       = "roles/artifactregistry.writer"
  member     = local.build_account
}

resource "terraform_data" "image" {
  triggers_replace = local.image
  input            = local.image

  provisioner "local-exec" {
    working_dir = local.repo_root
    command     = "gcloud builds submit --project ${var.project_id} --tag ${local.image} ."
  }

  depends_on = [
    google_project_iam_member.build_reads_source,
    google_project_iam_member.build_writes_logs,
    google_artifact_registry_repository_iam_member.build_pushes_image,
  ]
}
