"""
Version compatibility gating and file fingerprinting for the registry.

``SUPPORTED_CONVENTIONS`` is the single source of truth for which CODT output
conventions this version of codt_tools can read; ``SUPPORTED_INPUT_CONVENTIONS``
is the equivalent for the NetCDF *input* files a run is registered with. Bump
them via the codt-versioning workflow when the corresponding format changes.

Notes
-----
CODT v3 bumped the parcel input to ``CODT_parcel_input_v3`` (waypoint legs) but
left the output conventions at ``CODT_output_v1``. A reader therefore cannot use
the output conventions string to tell a v3 run from a v2 one — it must probe for
variable presence, or consult the registered ``parcel_input`` conventions.
"""

from __future__ import annotations

import hashlib
import shutil
import warnings
from pathlib import Path

import netCDF4 as nc

SUPPORTED_CONVENTIONS: frozenset[str] = frozenset({"CODT_output_v1"})

# Per-file-type input conventions this codt_tools can read/write. Parcel v1/v2
# (position-keyed segments) are retired: v3 re-keys the lookup to a leg counter
# and redefines ent_rate as 1/km, so an old file is silently wrong, not just
# unreadable.
SUPPORTED_INPUT_CONVENTIONS: dict[str, frozenset[str]] = {
    "parcel_input": frozenset({"CODT_parcel_input_v3"}),
    "aerosol_input": frozenset({"CODT_aerosol_input_v1"}),
}

# Conventions strings that were valid once and are now refused, with the reason.
RETIRED_INPUT_CONVENTIONS: dict[str, str] = {
    "CODT_parcel_input_v1": (
        "v1/v2 parcel inputs are position-keyed segments; v3 uses waypoint legs "
        "and redefines ent_rate as 1/km (a 1000x change). Regenerate the file — "
        "see docs/codt_v3_migration.md."
    ),
}
RETIRED_INPUT_CONVENTIONS["CODT_parcel_input_v2"] = RETIRED_INPUT_CONVENTIONS[
    "CODT_parcel_input_v1"
]

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


def inspect_input_file(path: str | Path) -> tuple[str | None, bool]:
    """
    Read the conventions string and seed-group presence of a NetCDF input.

    Seed-group detection probes for the ``seed_bin`` dimension, mirroring
    CODT's own reader (``droplets.f90`` ``read_seed_group``, which returns
    early when that dimension is absent). The group is optional and
    all-or-nothing; a file either carries the full set or none of it.

    Parameters
    ----------
    path : str or Path
        NetCDF input file (aerosol or parcel).

    Returns
    -------
    conventions : str or None
        The ``conventions`` global attribute, or None if absent.
    has_seed_group : bool
        True if the file carries an aerosol seed group.
    """
    with nc.Dataset(str(path), "r") as ds:
        return getattr(ds, "conventions", None), "seed_bin" in ds.dimensions


def check_input_conventions(
    file_type: str, value: str | None, strict: bool = False
) -> bool:
    """
    Check an input file's conventions string against the supported set.

    Parameters
    ----------
    file_type : str
        Key into ``SUPPORTED_INPUT_CONVENTIONS`` (``'parcel_input'`` or
        ``'aerosol_input'``). Unknown types pass unchecked.
    value : str or None
        The file's ``conventions`` global attribute.
    strict : bool
        If True, raise on incompatibility instead of warning.

    Returns
    -------
    bool
        True if the conventions string is supported.

    Raises
    ------
    IncompatibleConventionsError
        If ``strict`` is True and the value is unsupported.
    """
    supported = SUPPORTED_INPUT_CONVENTIONS.get(file_type)
    if supported is None or value in supported:
        return True

    msg = (
        f"{file_type} conventions {value!r} not in supported set "
        f"{sorted(supported)}."
    )
    if value in RETIRED_INPUT_CONVENTIONS:
        msg = f"{msg} {RETIRED_INPUT_CONVENTIONS[value]}"
    if strict:
        raise IncompatibleConventionsError(msg)
    warnings.warn(msg, stacklevel=2)
    return False


def check_seeding_consistency(
    do_seeding: bool, has_seed_group: bool, strict: bool = False
) -> bool:
    """
    Check ``&MICROPHYSICS do_seeding`` against the aerosol file's seed group.

    ``do_seeding`` is the sole controller, so the gate is **one-way** (mirrors
    CODT ``droplets.f90``):

    - ``do_seeding=.true.`` with no seed group is **fatal** — there is nothing
      to seed. This returns False (warns, or raises under ``strict``).
    - ``do_seeding=.false.`` with a seed group present is **fine**: CODT ignores
      the group (never reads it) and merely prints a warning, which is what lets
      one file serve both a seeded and an unseeded run. This returns True.

    Unlike the equivalent check in ``CODTConfig.validate()``, this inspects the
    file actually staged in the run directory — catching a shared or symlinked
    aerosol input swapped after the config was built.

    Parameters
    ----------
    do_seeding : bool
        The namelist flag as registered.
    has_seed_group : bool
        Whether the staged aerosol input carries a seed group.
    strict : bool
        If True, raise on the fatal case instead of warning.

    Returns
    -------
    bool
        True unless ``do_seeding`` is set with no seed group to act on.

    Raises
    ------
    IncompatibleConventionsError
        If ``strict`` is True and ``do_seeding`` is set with no seed group.
    """
    if not do_seeding or has_seed_group:
        return True

    msg = (
        "do_seeding=.true. but the staged aerosol input has no seed group; "
        "CODT will abort this run."
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
