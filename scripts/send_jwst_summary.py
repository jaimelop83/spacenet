#!/usr/bin/env python3
import os
import subprocess
import sys
from email.message import EmailMessage
import smtplib


def run_cmd(cmd):
    return subprocess.check_output(cmd, text=True).strip()


def main():
    to_addr = os.environ.get("JWST_EMAIL_TO")
    from_addr = os.environ.get("JWST_EMAIL_FROM")
    smtp_host = os.environ.get("JWST_SMTP_HOST")
    smtp_port = int(os.environ.get("JWST_SMTP_PORT", "587"))
    smtp_user = os.environ.get("JWST_SMTP_USER")
    smtp_pass = os.environ.get("JWST_SMTP_PASS")

    missing = [k for k in ["JWST_EMAIL_TO", "JWST_EMAIL_FROM", "JWST_SMTP_HOST"] if not os.environ.get(k)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    summary = run_cmd(["bash", "/home/jaimelop/spacenet/scripts/check_jwst_progress.sh"])

    msg = EmailMessage()
    msg["Subject"] = "JWST FITS Download Summary"
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(summary)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        if smtp_user and smtp_pass:
            server.login(smtp_user, smtp_pass)
        server.send_message(msg)

    print("Sent JWST summary email.")


if __name__ == "__main__":
    main()
