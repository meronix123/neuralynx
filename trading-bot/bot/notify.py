import logging

import requests

log = logging.getLogger("bot")


class Notifier:
    def __init__(self, token: str, chat_id: str):
        self.token, self.chat_id = token, chat_id

    def send(self, text: str) -> None:
        log.info(text)
        if not (self.token and self.chat_id):
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=10,
            )
        except requests.RequestException as e:
            # nicht die ganze Meldung loggen - die URL enthaelt den Telegram-Schluessel
            log.warning("Telegram fehlgeschlagen: %s", type(e).__name__)
