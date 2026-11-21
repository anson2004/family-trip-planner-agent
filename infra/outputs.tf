output "run_now" {
  description = "Start the job once by hand"
  value       = "gcloud run jobs execute ${google_cloud_run_v2_job.planner.name} --region ${var.region} --project ${var.project_id}"
}

output "logs" {
  description = "Where to read what the last runs printed"
  value       = "https://console.cloud.google.com/run/jobs/details/${var.region}/${google_cloud_run_v2_job.planner.name}/executions?project=${var.project_id}"
}

output "github_variables" {
  description = "Repository variables for the build and deploy jobs in pipeline.yml (identifiers, not secrets)"
  value = {
    GCP_PROJECT_ID   = var.project_id
    GCP_REGION       = var.region
    GCP_WIF_PROVIDER = google_iam_workload_identity_pool_provider.github.name
    GCP_BUILD_SA     = google_service_account.github_build.email
    GCP_DEPLOY_SA    = google_service_account.github_deploy.email
  }
}
