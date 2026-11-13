#!/usr/bin/env python3
"""Verify SHA-256 digests of every file recorded in ``MANIFEST_SHA256.json``."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "MANIFEST_SHA256.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not MANIFEST.exists():
        sys.exit(f"Missing {MANIFEST}")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = data["files"]
    failures = []
    for rel, meta in entries.items():
        p = REPO_ROOT / rel
        if not p.exists():
            failures.append(f"MISSING: {rel}")
            continue
        actual_sz = p.stat().st_size
        actual_sha = sha256_file(p)
        if actual_sz != meta["bytes"] or actual_sha != meta["sha256"]:
            failures.append(f"MISMATCH: {rel}")
    if failures:
        for msg in failures[:20]:
            print(msg, file=sys.stderr)
        sys.exit(f"Verification failed ({len(failures)} errors out of {len(entries)} files).")
    print(f"[PASS] Verified SHA-256 checksums for all {len(entries)} package files ({data['total_bytes']:,} bytes).")


if __name__ == "__main__":
    main()
