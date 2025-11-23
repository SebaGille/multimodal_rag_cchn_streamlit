#!/usr/bin/env python3
"""
Ensure every package inside the current virtual environment uses the same
OpenMP runtime. Torch bundles its own ``libomp.dylib`` while ``faiss-cpu``
ships another copy under ``faiss/.dylibs``. Loading both copies at runtime
triggers ``OMP: Error #15`` on macOS. The fix is to replace Torch's copy
with a symlink to the FAISS one so the dynamic loader resolves a single file.

Usage:
    python scripts/unify_openmp.py

Run this after installing requirements (and anytime pip reinstalls torch or
faiss). The script is idempotent and will skip work if the link already exists.
"""

from __future__ import annotations

import argparse
import sys
import sysconfig
from pathlib import Path


def _site_packages() -> Path:
    """Return the active site-packages path for the running interpreter."""
    return Path(sysconfig.get_paths()["purelib"]).resolve()


def _ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise SystemExit(f"{description} not found at {path}. Activate the .venv and reinstall deps.")


def _link_libomp(force: bool = False) -> None:
    site_packages = _site_packages()
    faiss_lib = site_packages / "faiss" / ".dylibs" / "libomp.dylib"
    torch_lib = site_packages / "torch" / "lib" / "libomp.dylib"

    _ensure_exists(faiss_lib, "FAISS libomp")
    if not torch_lib.exists():
        raise SystemExit(
            f"Torch libomp not found at {torch_lib}. Ensure torch is installed before running this script."
        )

    faiss_target = faiss_lib.resolve()

    if torch_lib.is_symlink() and torch_lib.resolve() == faiss_target:
        print(f"[ok] torch already links to {faiss_target}")
        return

    backup = torch_lib.with_suffix(torch_lib.suffix + ".bak")
    if torch_lib.exists() and not torch_lib.is_symlink():
        if backup.exists() and not force:
            raise SystemExit(f"Backup {backup} already exists. Use --force to overwrite.")
        torch_lib.rename(backup)
        print(f"[info] moved torch libomp to {backup}")
    elif torch_lib.is_symlink():
        torch_lib.unlink()

    if torch_lib.exists():
        if not force and not torch_lib.is_symlink():
            raise SystemExit(f"{torch_lib} still exists and is not a symlink. Use --force to overwrite.")
        torch_lib.unlink()

    torch_lib.symlink_to(faiss_target)
    print(f"[ok] linked {torch_lib} -> {faiss_target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Point torch at the FAISS libomp runtime.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite any existing backup/symlink without prompting.",
    )
    args = parser.parse_args()
    _link_libomp(force=args.force)


if __name__ == "__main__":
    if sys.platform != "darwin":
        raise SystemExit("This helper is only required on macOS hosts.")
    main()

