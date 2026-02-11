import os
import requests

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

class APIError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

def _handle_response(r: requests.Response):
    try:
        data = r.json()
    except Exception:
        data = None

    if not r.ok:
        msg = None
        if isinstance(data, dict):
            msg = data.get("detail") or data.get("message")
        msg = msg or f"Request failed ({r.status_code})"
        raise APIError(msg, r.status_code)

    return data

def post(path: str, payload: dict, token: str | None = None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(f"{API_BASE}{path}", json=payload, headers=headers, timeout=30)
    return _handle_response(r)

def get(path: str, token: str | None = None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"{API_BASE}{path}", headers=headers, timeout=30)
    return _handle_response(r)
