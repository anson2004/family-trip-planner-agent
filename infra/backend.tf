# State lives in the <project-id>-tfstate bucket (made by bootstrap-state.sh), shared by your Mac and
# GitHub Actions. The bucket name is passed at init: deploy.sh does that for you.
terraform {
  backend "gcs" {
    prefix = "travel-plan"
  }
}
