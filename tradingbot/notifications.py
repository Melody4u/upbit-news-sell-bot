import time
import logging
import requests


def notify_event(webhook_url: str, enabled_events: set, event_name: str, message: str) -> None:
    if not webhook_url or event_name not in enabled_events:
        return
    for attempt in range(2):
        try:
            requests.post(webhook_url, json={"event": event_name, "text": message}, timeout=5)
            return
        except Exception as e:
            if attempt == 1:
                logging.warning("notify failed(%s): %s", event_name, e)
            else:
                time.sleep(0.4)
