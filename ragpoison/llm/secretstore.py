"""API-key storage and resolution.

Three storage tiers, in decreasing order of preference (decision D24):

1. **DPAPI ciphertext** -- ``.secrets/deepseek.dpapi``. Encrypted with
   ``CryptProtectData`` under the current Windows user, so the plaintext never
   touches the environment block, the shell history, or the repository. This is
   the default and the recommended tier.
2. **``.env``** -- a git-ignored file in the repository root. Read explicitly by
   this module; it is never exported into the process environment by us.
3. **``DEEPSEEK_API_KEY``** -- supported for CI or containers, but the variable
   is removed from ``os.environ`` immediately after it is read so that child
   processes and traceback dumping cannot observe it.

``setx`` is deliberately unsupported: it writes the key in plaintext to
``HKCU\\Environment`` and to the PowerShell history file, and every later process
of the same user inherits it.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path

from ..paths import path as repo_path
from ..redact import mask_key

ENV_VAR = "DEEPSEEK_API_KEY"
SECRETS_DIR: Path = repo_path(".secrets")
DPAPI_FILE: Path = SECRETS_DIR / "deepseek.dpapi"
ENV_FILE: Path = repo_path(".env")
_KEY_DESCRIPTION = "ragpoison-deepseek-key"

IS_WINDOWS: bool = sys.platform == "win32"

# CRYPTPROTECT_UI_FORBIDDEN: never show an interactive prompt.
_CRYPTPROTECT_UI_FORBIDDEN = 0x01
_CRYPTPROTECT_LOCAL_MACHINE = 0x04  # intentionally NOT used: per-user scope only


# --------------------------------------------------------------------------- #
# DPAPI (Windows) via ctypes -- no third-party dependency
# --------------------------------------------------------------------------- #


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _crypt32():  # pragma: no cover - platform shim
    if not IS_WINDOWS:
        raise RuntimeError("DPAPI is only available on Windows; use the .env tier instead")
    return ctypes.windll.crypt32


def _kernel32():  # pragma: no cover - platform shim
    return ctypes.windll.kernel32


def _to_blob(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
    """Build a DATA_BLOB plus the buffer that keeps it alive."""
    buffer = ctypes.create_string_buffer(data, len(data))
    blob = _DataBlob(
        cbData=len(data),
        pbData=ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)),
    )
    return blob, buffer


def _from_blob(blob: _DataBlob) -> bytes:
    return ctypes.string_at(blob.pbData, blob.cbData)


def _free(pointer) -> None:
    _kernel32().LocalFree(ctypes.cast(pointer, ctypes.c_void_p))


def dpapi_protect(plaintext: bytes) -> bytes:
    """Encrypt ``plaintext`` for the current Windows user and machine."""
    crypt32 = _crypt32()
    in_blob, _keep = _to_blob(plaintext)
    out_blob = _DataBlob()
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    ok = crypt32.CryptProtectData(
        ctypes.byref(in_blob),
        _KEY_DESCRIPTION,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise OSError(ctypes.GetLastError(), "CryptProtectData failed")
    try:
        return _from_blob(out_blob)
    finally:
        _free(out_blob.pbData)


def dpapi_unprotect(ciphertext: bytes) -> bytes:
    """Decrypt a blob produced by :func:`dpapi_protect` on this user account."""
    crypt32 = _crypt32()
    in_blob, _keep = _to_blob(ciphertext)
    out_blob = _DataBlob()
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(out_blob),
    )
    if not ok:
        raise OSError(
            ctypes.GetLastError(),
            "CryptUnprotectData failed (different Windows user or machine?)",
        )
    try:
        return _from_blob(out_blob)
    finally:
        _free(out_blob.pbData)


# --------------------------------------------------------------------------- #
# .env tier
# --------------------------------------------------------------------------- #


def _read_dotenv(env_file: Path = ENV_FILE) -> str | None:
    """Return ``DEEPSEEK_API_KEY`` from ``.env`` without touching the environment."""
    if not env_file.exists():
        return None
    try:
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            name, _, value = line.partition("=")
            if name.strip() != ENV_VAR:
                continue
            return value.strip().strip('"').strip("'") or None
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _normalise(key: str) -> str:
    key = key.strip().strip('"').strip("'")
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def set_key(key: str) -> str:
    """Store ``key`` in the DPAPI tier and return the storage label."""
    key = _normalise(key)
    if not key:
        raise ValueError("empty API key")
    if not key.startswith("sk-"):
        raise ValueError("unexpected key format: a DeepSeek key starts with 'sk-'")
    if len(key) < 20:
        raise ValueError("unexpected key format: the key is too short")
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    DPAPI_FILE.write_bytes(dpapi_protect(key.encode("utf-8")))
    return "dpapi"


def clear_stored() -> bool:
    """Delete the DPAPI ciphertext. Returns True when a file was removed."""
    if DPAPI_FILE.exists():
        for _ in range(3):  # best effort; AV scanners occasionally hold the file
            try:
                DPAPI_FILE.unlink()
                return True
            except OSError:
                continue
        return False
    return False


def read_key(required: bool = False) -> tuple[str | None, str]:
    """Resolve the key. Returns ``(key, source)`` where source is one of
    ``dpapi``, ``dotenv``, ``env`` or ``none``.

    The ``DEEPSEEK_API_KEY`` variable is removed from ``os.environ`` as soon as
    it has been read, whether or not it was the winning tier.
    """
    key: str | None = None
    source = "none"

    if DPAPI_FILE.exists():
        try:
            key = dpapi_unprotect(DPAPI_FILE.read_bytes()).decode("utf-8").strip()
            source = "dpapi"
        except OSError:
            key = None

    if not key:
        key = _read_dotenv()
        if key:
            source = "dotenv"

    env_value = os.environ.pop(ENV_VAR, None)
    if not key and env_value:
        key = _normalise(env_value)
        source = "env"

    if required and not key:
        raise RuntimeError(
            "no DeepSeek API key found. Run:  python run.py secret set\n"
            "  tiers: .secrets/deepseek.dpapi > .env > DEEPSEEK_API_KEY"
        )
    return key, source


def status() -> dict[str, object]:
    """Describe where the key comes from, without revealing it."""
    key, source = read_key()
    notes: list[str] = []
    if DPAPI_FILE.exists() and _read_dotenv():
        notes.append("both DPAPI and .env present; DPAPI wins")
    if IS_WINDOWS and not DPAPI_FILE.exists():
        notes.append("no DPAPI ciphertext yet; run `python run.py secret set`")
    if not IS_WINDOWS:
        notes.append("non-Windows host: DPAPI unavailable, use the .env tier")
    return {
        "source": source,
        "present": key is not None,
        "masked": mask_key(key),
        "length": len(key) if key else 0,
        "dpapi_file": str(DPAPI_FILE.relative_to(repo_path())) if DPAPI_FILE.exists() else "",
        "dotenv_file": str(ENV_FILE.name) if ENV_FILE.exists() else "",
        "notes": notes,
    }


def verify_key(key: str | None = None, timeout_s: float = 20.0) -> dict[str, object]:
    """Validate the key against the free ``GET /user/balance`` endpoint.

    This endpoint does not consume tokens. Network or endpoint problems are
    reported as ``ok=False`` with a message rather than raised, because the key
    is already stored successfully at that point.
    """
    if key is None:
        key, _source = read_key(required=True)

    try:
        import requests
    except ImportError:  # pragma: no cover - requests is a hard requirement
        return {"ok": False, "status": None, "balance": None, "message": "requests not installed"}

    url = "https://api.deepseek.com/user/balance"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - any transport error is reported, not raised
        return {
            "ok": False,
            "status": None,
            "balance": None,
            "message": f"could not reach the API ({type(exc).__name__}); key stored but unverified",
        }

    if response.status_code == 401:
        return {"ok": False, "status": 401, "balance": None, "message": "key rejected (401)"}
    if response.status_code != 200:
        return {
            "ok": False,
            "status": response.status_code,
            "balance": None,
            "message": f"unexpected status {response.status_code}; key stored but unverified",
        }

    balance = None
    try:
        payload = response.json()
        infos = payload.get("balance_infos") or []
        if infos:
            first = infos[0]
            balance = f"{first.get('total_balance')} {first.get('currency', '')}".strip()
        elif payload.get("is_available") is not None:
            balance = "available" if payload["is_available"] else "unavailable"
    except ValueError:
        balance = None

    return {"ok": True, "status": 200, "balance": balance, "message": "ok"}
