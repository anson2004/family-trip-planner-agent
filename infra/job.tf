# The Cloud Run Job that runs `travel-plan` once, and the weekly trigger that starts it.

# The job's own identity: it may only read its API keys.
resource "google_service_account" "job" {
  account_id   = "travel-plan-job"
  display_name = "Travel plan job"
  depends_on   = [google_project_service.apis]
}

locals {
  secret_ids = { for name in var.secrets : name => lower(replace(name, "_", "-")) }
}

resource "google_secret_manager_secret_iam_member" "job_reads_keys" {
  for_each  = local.secret_ids
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.job.email}"
}

resource "google_cloud_run_v2_job" "planner" {
  name                = "travel-plan"
  location            = var.region
  deletion_protection = false

  template {
    task_count = 1
    template {
      service_account = google_service_account.job.email
      timeout         = "900s"
      # A retry would repeat every search and spend the SerpApi quota again.
      max_retries = 0

      containers {
        image = var.image
        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }

        # The names (TRIP_TO, EMAIL_TO...) aren't personal; only the values stay hidden in plans.
        dynamic "env" {
          for_each = nonsensitive(toset(keys(var.settings)))
          content {
            name  = env.value
            value = var.settings[env.value]
          }
        }

        dynamic "env" {
          for_each = local.secret_ids
          content {
            name = env.key
            value_source {
              secret_key_ref {
                secret  = env.value
                version = "latest"
              }
            }
          }
        }
      }
    }
  }

  depends_on = [google_secret_manager_secret_iam_member.job_reads_keys]
}

# Cloud Scheduler starts the job with its own identity, which may only run this job.
resource "google_service_account" "scheduler" {
  account_id   = "travel-plan-scheduler"
  display_name = "Travel plan scheduler"
  depends_on   = [google_project_service.apis]
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_runs_job" {
  name     = google_cloud_run_v2_job.planner.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "weekly" {
  name      = "travel-plan-weekly"
  region    = var.scheduler_region
  schedule  = var.schedule
  time_zone = var.time_zone

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.planner.name}:run"
    oauth_token {
      service_account_email = google_service_account.scheduler.email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  depends_on = [google_cloud_run_v2_job_iam_member.scheduler_runs_job]
}
