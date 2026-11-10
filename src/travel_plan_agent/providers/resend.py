"""Send email through Resend (https://resend.com/docs/api-reference/emails/send-email)."""

import httpx

from travel_plan_agent.email_report import EmailContent

RESEND_URL = "https://api.resend.com/emails"


class ResendError(RuntimeError):
    pass


def send_email(
    api_key: str, sender: str, to: list[str], content: EmailContent, http: httpx.Client | None = None
) -> str:
    """Send the email and return Resend's message ID."""
    client = http or httpx.Client(timeout=30)
    response = client.post(
        RESEND_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"from": sender, "to": to, "subject": content.subject, "html": content.html, "text": content.text},
    )
    if response.status_code >= 400:
        try:
            message = response.json().get("message", response.text)
        except ValueError:
            message = response.text
        raise ResendError(f"Resend returned HTTP {response.status_code}: {message}")
    email_id: str = response.json().get("id", "")
    return email_id
