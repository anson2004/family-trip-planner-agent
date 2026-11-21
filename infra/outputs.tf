output "run_now" {
  description = "Start the job once by hand"
  value       = "gcloud run jobs execute ${google_cloud_run_v2_job.planner.name} --region ${var.region} --project ${var.project_id}"
}

output "logs" {
  description = "Where to read what the last runs printed"
  value       = "https://console.cloud.google.com/run/jobs/details/${var.region}/${google_cloud_run_v2_job.planner.name}/executions?project=${var.project_id}"
}

output "image" {
  value = terraform_data.image.output
}
