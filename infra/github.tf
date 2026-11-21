# Lets GitHub Actions in var.github_repo log in to Google Cloud without any stored key
# (Workload Identity Federation). Only workflows running on the master branch are accepted.

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  display_name                       = "GitHub"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }
  attribute_condition = "assertion.repository == \"${var.github_repo}\" && assertion.ref == \"refs/heads/master\""
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

locals {
  github_principal = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
  state_bucket     = "${var.project_id}-tfstate"
}

# Build job: may only push images to the travel-plan repository.
resource "google_service_account" "github_build" {
  account_id   = "github-build"
  display_name = "GitHub Actions: build images"
  depends_on   = [google_project_service.apis]
}

resource "google_artifact_registry_repository_iam_member" "github_build_pushes" {
  repository = google_artifact_registry_repository.app.name
  location   = var.region
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.github_build.email}"
}

resource "google_service_account_iam_member" "github_build_login" {
  service_account_id = google_service_account.github_build.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.github_principal
}

# Deploy job: runs `tofu apply`, so it needs rights over everything this folder manages,
# plus the state and config in the bucket. Only master of var.github_repo can use it.
resource "google_service_account" "github_deploy" {
  account_id   = "github-deploy"
  display_name = "GitHub Actions: deploy with OpenTofu"
  depends_on   = [google_project_service.apis]
}

resource "google_project_iam_member" "github_deploy" {
  for_each = toset([
    "roles/artifactregistry.admin",
    "roles/cloudscheduler.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/iam.serviceAccountUser",
    "roles/iam.workloadIdentityPoolAdmin",
    "roles/resourcemanager.projectIamAdmin",
    "roles/run.admin",
    "roles/secretmanager.admin",
    "roles/serviceusage.serviceUsageAdmin",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.github_deploy.email}"
}

# Admin of this one bucket only: it reads and writes the state and config, and manages this grant itself.
resource "google_storage_bucket_iam_member" "github_deploy_state" {
  bucket = local.state_bucket
  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.github_deploy.email}"
}

resource "google_service_account_iam_member" "github_deploy_login" {
  service_account_id = google_service_account.github_deploy.name
  role               = "roles/iam.workloadIdentityUser"
  member             = local.github_principal
}
