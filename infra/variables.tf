variable "project_id" {
  description = "Google Cloud project ID, e.g. travel-plan-agent-123456"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository (owner/name) whose master branch may build and deploy"
  type        = string
}

variable "image" {
  description = "Container image to run, e.g. <repo>/app@sha256:... (deploy.sh passes the newest one)"
  type        = string
}

variable "region" {
  description = "Where the job and its images live"
  type        = string
  default     = "europe-north1" # Finland
}

variable "scheduler_region" {
  description = "Where the Cloud Scheduler trigger lives (not every region has Cloud Scheduler)"
  type        = string
  default     = "europe-west1"
}

variable "schedule" {
  description = "When to run, in cron format"
  type        = string
  default     = "0 8 * * 1" # Mondays 08:00
}

variable "time_zone" {
  type    = string
  default = "Europe/Helsinki"
}

variable "settings" {
  description = "Non-secret settings for the job: TRIP_*, EMAIL_TO, GROQ_MODEL... (same names as in .env)"
  type        = map(string)
}

variable "secrets" {
  description = "API keys stored in Secret Manager by set-secrets.sh. Leave out the ones you don't use."
  type        = list(string)
  default     = ["GROQ_API_KEY", "SERPAPI_API_KEY", "TAVILY_API_KEY", "RESEND_API_KEY"]
}
