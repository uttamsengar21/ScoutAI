"""Send the digest over SMTP. Gmail: use an App Password, not your login."""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def send(subject: str, html_body: str) -> None:
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "465"))
    user = (os.environ.get("SMTP_USER") or "").strip()
    password = (os.environ.get("SMTP_PASS") or "").strip().replace(" ", "")
    to_addr = (os.getenv("MAIL_TO", user) or "").strip()

    if not user or not password:
        raise ValueError("SMTP_USER or SMTP_PASS is missing or blank")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content("This digest is HTML. Open it in an HTML-capable client.")
    msg.add_alternative(html_body, subtype="html")

    # Try preferred port first (465 SSL or 587 STARTTLS), then auto-fallback if connection is closed
    ports_to_try = [465, 587] if port == 465 else [587, 465]
    last_err: Exception | None = None

    for p in ports_to_try:
        try:
            if p == 465:
                # Direct SSL (works on most cloud providers)
                with smtplib.SMTP_SSL(host, p, timeout=25) as s:
                    s.login(user, password)
                    s.send_message(msg)
            else:
                # STARTTLS (RFC standard)
                with smtplib.SMTP(host, p, timeout=25) as s:
                    s.ehlo()
                    s.starttls()
                    s.ehlo()
                    s.login(user, password)
                    s.send_message(msg)
            print(f"  mailed -> {to_addr} (via port {p})")
            return
        except Exception as e:
            last_err = e

    if last_err:
        raise last_err
