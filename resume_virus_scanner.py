"""
resume_virus_scanner.py

Industry-standard virus scanning for uploaded resume files (PDF and images)

Engine   : ClamAV via clamd daemon (TCP or Unix socket)
           - Free, open-source, used in production by AWS, Cloudflare, etc.
           - Scans via INSTREAM — no filesystem access to the file required
             (works in containerised / serverless environments)
           Install:
             Ubuntu/Debian : sudo apt install clamav clamav-daemon
             macOS         : brew install clamav
             Windows       : https://www.clamav.net/downloads
           Python binding  : pip install clamd

Fail-closed policy:
    If ClamAV is unavailable, the file is REJECTED (not silently passed)
    This prevents an unscanned file from entering the NLP pipeline

Error codes (aligned with existing pipeline):
    800 : File not found before scan
    801 : File type not permitted
    802 : File exceeds scan size limit (> 25 MB)
    803 : ClamAV — threat detected
    804 : ClamAV unavailable (fail-closed)
"""

import hashlib
import json
import os
import sys

# Constants
MAX_FILE_BYTES = 25 * 1024 * 1024 # 25 MB hard limit
CLAMAV_HOST    = os.environ.get("CLAMAV_HOST", "127.0.0.1")
CLAMAV_PORT    = int(os.environ.get("CLAMAV_PORT", "3310"))
CLAMAV_SOCKET  = os.environ.get("CLAMAV_SOCKET", None)
CLAMAV_TIMEOUT = int(os.environ.get("CLAMAV_TIMEOUT", "60"))

PERMITTED_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"
}

class FileExceedsSizeError(Exception):
    pass

class FileTypeNotPermittedError(Exception):
    pass

class VirusDetectedError(Exception):
    pass

class VirusScannerUnavailableError(Exception):
    pass

# Internal helpers
def _check_file_prerequisites(file_path: str) -> None:
    """
    Pre-scan gate checks - runs before any AV engine is invoked:
      1. File exists on disk
      2. Extension is in the permitted set
      3. File size is within the 25 MB limit
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError
    except FileNotFoundError:
        error_response = {
            "status" : "error",
            "message": f"File not found before virus scan: {file_path}",
            "code"   : 800
        }
        print(json.dumps(error_response))
        sys.exit()

    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in PERMITTED_EXTENSIONS:
            raise FileTypeNotPermittedError
    except FileTypeNotPermittedError:
        error_response = {
            "status" : "error",
            "message": f"File type '{ext}' is not permitted. Allowed: {', '.join(sorted(PERMITTED_EXTENSIONS))}",
            "code"   : 801
        }
        print(json.dumps(error_response))
        sys.exit()

    try:
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_BYTES:
            raise FileExceedsSizeError
    except FileExceedsSizeError:
        error_response = {
            "status" : "error",
            "message": f"File exceeds the 25 MB virus-scan limit ({file_size / 1_048_576:.1f} MB): {file_path}",
            "code"   : 802
        }
        print(json.dumps(error_response))
        sys.exit()

# ClamAV (sole engine)
def _resolve_clamav_socket() -> str | None:
    """
    Return the first auto-detected ClamAV Unix socket path, or None.

    Common locations across distros:
      /var/run/clamav/clamd.sock   — Debian / Ubuntu
      /tmp/clamd.socket            — macOS Homebrew
      /run/clamav/clamd.sock       — Fedora / RHEL
      /var/run/clamd.scan/clamd.sock    — Fedora RPM (clamd@scan)
    An explicit CLAMAV_SOCKET env var always takes priority.
    """
    if CLAMAV_SOCKET:
        return CLAMAV_SOCKET
    for candidate in (
        "/var/run/clamav/clamd.sock",
        "/tmp/clamd.socket",
        "/run/clamav/clamd.sock",
        "/var/run/clamd.scan/clamd.sock",
    ):
        if os.path.exists(candidate):
            return candidate
    return None

def _scan_with_clamav(file_path: str) -> tuple[bool, bool]:
    """
    Stream the file to the ClamAV daemon using INSTREAM scanning

    INSTREAM sends raw bytes over the socket — clamd never needs filesystem
    access to the file, which is important in containerised environments
    """
    try:
        import clamd
    except ImportError:
        print(
            "[ClamAV] 'clamd' library not installed (pip install clamd)."
            "Skipping ClamAV."
        )
        return False, True

    try:
        socket_path = _resolve_clamav_socket()
        if socket_path:
            print(f"[ClamAV]: connecting via Unix socket: {socket_path}")
            cd = clamd.ClamdUnixSocket(path = socket_path, timeout = CLAMAV_TIMEOUT)
        else:
            print(f"[ClamAV]: connecting via TCP {CLAMAV_HOST}:{CLAMAV_PORT}")
            cd = clamd.ClamdNetworkSocket(
                host = CLAMAV_HOST, port = CLAMAV_PORT, timeout = CLAMAV_TIMEOUT
            )
        cd.ping() # raises ConnectionError if daemon down
    except Exception as e:
        print(f"[ClamAV] daemon not reachable: {e}")
        return False, True

    try:
        print(f"[ClamAV] scanning '{os.path.basename(file_path)}' ...")
        with open(file_path, "rb") as fh:
            result = cd.instream(fh) # {"stream": ("OK"|"FOUND", [sig])}

        status, *detail = result.get("stream", ("UNKNOWN",))

        if status == "OK":
            print("[ClamAV] CLEAN.")
            return True, True

        threat = detail[0] if detail else "unknown threat"
        print(f"[ClamAV] THREAT DETECTED — {threat}")
        return True, False

    except Exception as e:
        print(f"[ClamAV] scan error: {e}")
        return False, True # treat scan error as unavailable

def _sha256(file_path: str) -> str:
    """Return the SHA-256 hex digest of a file (used as audit trail only)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# Public API — the only symbol callers should import
def scan_file_for_viruses(file_path: str) -> None:
    """
    Virus-scan a file using ClamAV (local, no third-party transmission)
    """
    print(f"Starting file pre-scan: '{os.path.basename(file_path)}' ...")

    _check_file_prerequisites(file_path)

    size_mb = os.path.getsize(file_path) / 1_048_576
    print(f"File size: {size_mb:.2f} MB")

    # ClamAV
    clamav_available, clamav_clean = _scan_with_clamav(file_path)

    try:
        if clamav_available:
            if not clamav_clean:
                raise VirusDetectedError
            print("[ClamAV] scan passed. Proceeding.")
            return
        else:
            raise VirusScannerUnavailableError
    except VirusDetectedError:
        error_response = {
            "status" : "error",
            "message": f"Virus scan (ClamAV) THREAT DETECTED in: '{file_path}'. The file has been rejected.",
            "code"   : 803
        }
        print(json.dumps(error_response))
        sys.exit()
    except VirusScannerUnavailableError:
        error_response = {
            "status" : "error",
            "message": "Virus scan could not be completed: ClamAV is unavailable. Install ClamAV (https://docs.clamav.net) and ensure the daemon is running. File rejected for safety.",
            "code"   : 804
        }
        print(json.dumps(error_response))
        sys.exit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3.12 resume_virus_scanner.py <file_path>")
        sys.exit()

    scan_file_for_viruses(sys.argv[1])
    print(f"[ClamAV] '{sys.argv[1]}' is clean. Safe to process.")
