"""
netcheck.py — can this machine reach what the CRM import needs?

    python -m core.netcheck

Checks the CRM database port (DB_HOST / DB_PORT from .env, connection only, no
login or query), the CRM API, and the bizRTC recording server, and prints this
machine's public IP (what an allow-list needs). Works on Windows and Linux.
"""
from __future__ import annotations

import socket

import requests

from config import CRM_API_URL, DB_HOST, DB_PORT

BIZRTC = "https://smrweb.bizrtc.com/"


def _db() -> str:
    if not DB_HOST:
        return "skipped (DB_HOST not set in .env)"
    try:
        socket.create_connection((DB_HOST, DB_PORT), timeout=8).close()
        return "reachable"
    except OSError as exc:
        return f"NOT reachable ({type(exc).__name__})"


def _https(url: str, verify=True) -> str:
    try:
        return f"reachable (HTTP {requests.get(url, timeout=10, verify=verify).status_code})"
    except requests.exceptions.SSLError:
        return "reachable, but its certificate could not be verified"
    except requests.RequestException as exc:
        return f"NOT reachable ({type(exc).__name__})"


if __name__ == "__main__":
    from core.recordings import _ca_bundle

    print(f"CRM database  : {_db()}")
    print(f"CRM API       : {_https(CRM_API_URL.rstrip('/') + '/v3/api-docs')}")
    print(f"bizRTC        : {_https(BIZRTC, verify=_ca_bundle())}")
    try:
        print(f"public IP     : {requests.get('https://ifconfig.me/ip', timeout=10).text.strip()}")
    except requests.RequestException as exc:
        print(f"public IP     : unknown ({type(exc).__name__})")
