"""Python shim for the BUSHWHACKER Rust CLI.

The ``bushwhacker`` wheels bundle the compiled Rust binary. This package
exists so that ``python -m bushwhacker`` works and so downstream Python
code can locate the binary via :func:`binary_path`. All inference happens
in the Rust engine; there is no Python implementation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bushwhacker")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+unknown"


def binary_path() -> str:
    """Return the path to the installed ``bushwhacker`` binary."""
    found = shutil.which("bushwhacker")
    if found is None:
        # Wheels install the binary into the environment's bin/Scripts dir,
        # which is not guaranteed to be on PATH (e.g. pip --target installs
        # or launchers with a sanitized environment).
        candidates = (
            os.path.join(sys.exec_prefix, "bin", "bushwhacker"),
            os.path.join(sys.exec_prefix, "Scripts", "bushwhacker.exe"),
        )
        found = next((c for c in candidates if os.path.exists(c)), None)
    if found is None:
        raise FileNotFoundError(
            "Could not locate the 'bushwhacker' binary. If you installed "
            "from source, run `cargo build --release` or reinstall with "
            "`pip install --force-reinstall bushwhacker`."
        )
    return found


def main(argv: list[str] | None = None) -> int:
    """Run the ``bushwhacker`` CLI, forwarding all arguments.

    Returns the CLI's exit code.
    """
    args = sys.argv[1:] if argv is None else argv
    return subprocess.run([binary_path(), *args], check=False).returncode
