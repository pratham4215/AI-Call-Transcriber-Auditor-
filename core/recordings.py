"""
recordings.py — call recordings from the Kataria CRM decrypt API, by CDR id.

    POST {CRM_API_URL}/api/v1/cdr/recording/decrypt?cdrId=<id>

The .bizrtc files on the PBX are encrypted; this endpoint has bizRTC decrypt
the call's recording and answers with a playable, downloadable link (JSON or
plain text), which is then downloaded without the API key. Audio in the
response itself is used as-is.

Check the API with one CDR id (prints status and format, never the audio, key or link):
    python -m core.recordings 12345678
    python -m core.recordings 12345678 --save test_recording
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple
from urllib.parse import urlsplit

import certifi
import requests

from config import CRM_API_TIMEOUT, RESOURCES_DIR
from core import crm_api

_CDR_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_CERTS_DIR = RESOURCES_DIR / "certs"


class RecordingError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _ca_bundle() -> str:
    """certifi's roots plus the intermediates in resources/certs.

    bizRTC's recording server (smrweb.bizrtc.com) sends its certificate without
    the Sectigo intermediate that issued it. Browsers fetch that themselves and
    Python does not, so the (public) intermediate is shipped instead of turning
    verification off.
    """
    pem = Path(certifi.where()).read_text(encoding="ascii")
    for extra in sorted(_CERTS_DIR.glob("*.pem")):
        pem += "\n" + extra.read_text(encoding="ascii")
    path = Path(tempfile.gettempdir()) / f"kataria-audit-ca-{hashlib.sha256(pem.encode()).hexdigest()[:12]}.pem"
    if not path.exists():
        path.write_text(pem, encoding="ascii")
    return str(path)


def _download(link: str, timeout: float = CRM_API_TIMEOUT) -> requests.Response:
    try:
        # Keep-alive session; never carries the CRM key (only crm_api.request adds it).
        return crm_api.session().get(link, timeout=timeout, verify=_ca_bundle())
    except requests.RequestException as exc:
        # The link's file name holds the customer's phone number: report the host only.
        raise RecordingError(
            f"Could not download the recording from {urlsplit(link).hostname}: {type(exc).__name__}"
        ) from None


def audio_kind(data: bytes) -> Optional[str]:
    """File extension for a known audio signature, else None."""
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "mp3"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"fLaC":
        return "flac"
    if data[4:8] == b"ftyp":
        return "m4a"
    return None


def _request_api(cdr_id: str) -> requests.Response:
    cdr_id = str(cdr_id).strip()
    if not _CDR_ID_RE.match(cdr_id):
        raise RecordingError(f"Invalid CDR id: {cdr_id!r}")
    return crm_api.request("POST", crm_api.DECRYPT, {"cdrId": cdr_id})


def _json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _links(value: Any) -> Iterator[str]:
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _links(item)
    elif isinstance(value, list):
        for item in value:
            yield from _links(item)


def _resolve(resp: requests.Response, timings: Optional[dict] = None) -> Tuple[bytes, str]:
    if resp.status_code != 200:
        raise RecordingError(f"Recording API returned HTTP {resp.status_code}: {crm_api.message(resp)}")
    kind = audio_kind(resp.content)
    if kind:
        return resp.content, kind
    body = _json(resp)
    if body is None:
        body = resp.text.strip().strip('"')  # bare link as plain text
    link = next(_links(body), None)
    if link:
        started = time.perf_counter()
        follow = _download(link)
        if timings is not None:
            timings["download"] = time.perf_counter() - started
        kind = audio_kind(follow.content) if follow.status_code == 200 else None
        if kind:
            return follow.content, kind
        raise RecordingError(f"Recording link returned HTTP {follow.status_code}, not audio")
    raise RecordingError(f"No recording link in the API response: {crm_api.message(resp)}")


def fetch_recording(cdr_id: str, timings: Optional[dict] = None) -> Tuple[bytes, str]:
    """(decrypted audio bytes, file extension) for one CDR id.

    timings, if given, gets the seconds spent in "decrypt" (CRM API) and "download" (bizRTC).
    """
    started = time.perf_counter()
    resp = _request_api(cdr_id)
    if timings is not None:
        timings["decrypt"] = time.perf_counter() - started
    return _resolve(resp, timings)


def describe(resp: requests.Response) -> str:
    """Status and response shape (no audio, key or link contents)."""
    kind = audio_kind(resp.content)
    lines = [
        f"HTTP {resp.status_code} {resp.reason}",
        f"content-type: {resp.headers.get('Content-Type')}",
        f"size: {len(resp.content):,} bytes",
        f"audio: {kind or 'no'}",
    ]
    if resp.status_code != 200:
        lines.append(f"message: {crm_api.message(resp)}")
    body = _json(resp)
    text = resp.text.strip().strip('"') if body is None and not kind else ""
    if body is not None:
        lines.append(f"json: {json.dumps(crm_api.shape(body))}")
    elif text.startswith(("http://", "https://")):
        lines.append(f"text: {crm_api.shape(text)}")
    elif not kind:
        lines.append(f"first bytes: {resp.content[:16].hex()}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check the recording API with one CDR id.")
    parser.add_argument("cdr_id")
    parser.add_argument("--save", metavar="NAME", help="save the decrypted audio as NAME.<ext>")
    args = parser.parse_args()

    response = _request_api(args.cdr_id)
    print(describe(response))
    if args.save:
        try:
            audio, ext = _resolve(response)
        except RecordingError as exc:
            raise SystemExit(f"error: {exc}")
        out = Path(f"{args.save}.{ext}")
        out.write_bytes(audio)
        print(f"saved {out} ({len(audio):,} bytes)")
