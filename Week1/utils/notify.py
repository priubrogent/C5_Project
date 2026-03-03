"""
Simple ntfy.sh push notifications for MCV C5 training jobs.

Usage:
    from utils.notify import notify
    notify("Training started", title="R-CNN Task E")
    notify("Training done! mAP=0.42", title="R-CNN Task E", priority="high")

Shell (bash scripts):
    curl -s -d "message" ntfy.sh/mcvC5
"""

import requests

NTFY_URL = "https://ntfy.sh/mcvC5"


def notify(message: str, title: str = "MCV C5", priority: str = "default", tags: list = None):
    """
    Send a push notification via ntfy.sh. Fails silently so training never breaks.

    Args:
        message:  Body text of the notification.
        title:    Notification title shown on the device.
        priority: "low", "default", "high", or "urgent".
        tags:     List of emoji/tag strings, e.g. ["white_check_mark"].
    """
    headers = {
        "Title":    title,
        "Priority": priority,
    }
    if tags:
        headers["Tags"] = ",".join(tags)

    try:
        requests.post(NTFY_URL, data=message.encode("utf-8"), headers=headers, timeout=5)
    except Exception:
        pass  # never block training on a notification failure
