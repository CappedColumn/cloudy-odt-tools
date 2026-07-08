"""
Version compatibility gating and file fingerprinting for the registry.

``SUPPORTED_CONVENTIONS`` is the single source of truth for which CODT output
conventions this version of codt_tools can read. Bump it via the
codt-versioning workflow when the output format changes.
"""

from __future__ import annotations

import hashlib
import shutil
import warnings
from pathlib import Path

SUPPORTED_CONVENTIONS: frozenset[str] = frozenset({"CODT_output_v1"})

# Bytes per read when hashing large files (netCDF inputs can be >100 MB).
_HASH_CHUNK_BYTES: int = 1 << 20


class IncompatibleConventionsError(RuntimeError):
    """Raised in strict mode when an output conventions string is unsupported."""


def check_conventions(value: str | None, strict: bool = False) -> bool:
    """
    Check a CODT output conventions string against supported versions.

    Parameters
    ----------
    value : str or None
        The ``conventions`` global attribute of an output file, or None if
        the attribute is missing.
    strict : bool
        If True, raise on incompatibility instead of warning.

    Returns
    -------
    bool
        True if the conventions string is supported.

    Raises
    ------
    IncompatibleConventionsError
        If ``strict`` is True and the value is unsupported or missing.
    """
    if value in SUPPORTED_CONVENTIONS:
        return True

    msg = (
        f"Output conventions {value!r} not in supported set "
        f"{sorted(SUPPORTED_CONVENTIONS)}; analysis tools may misread this "
        "file. Use a codt_tools version matching the run's conventions."
    )
    if strict:
        raise IncompatibleConventionsError(msg)
    warnings.warn(msg, stacklevel=2)
    return False


def sha256_file(path: str | Path) -> str:
    """
    Compute the SHA256 hex digest of a file's (resolved) content.

    Parameters
    ----------
    path : str or Path
        File to hash. Symlinks are followed.

    Returns
    -------
    str
        64-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def archive_executable(executable: str | Path, archive_dir: str | Path) -> Path:
    """
    Copy an executable into a content-addressed archive, deduplicating.

    The archive layout is ``{archive_dir}/{sha256[:16]}/{exe_name}``: one copy
    per unique binary regardless of how many runs reference it. Existing
    entries are never overwritten.

    Parameters
    ----------
    executable : str or Path
        Path to the binary that was (or will be) executed.
    archive_dir : str or Path
        Root of the executable archive.

    Returns
    -------
    Path
        Path of the archived copy.
    """
    executable = Path(executable).expanduser().resolve()
    checksum = sha256_file(executable)
    target_dir = Path(archive_dir).expanduser() / checksum[:16]
    target = target_dir / executable.name
    if not target.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(executable, target)
    return target
