"""
Simulation registry for CODT runs and experiments.

SQLite-backed tracking of experiments (hypothesis -> conclusion), runs,
input files, namelist parameters, and status history. See
``docs/registry-quickstart.md`` for usage.
"""

from codt_tools.registry.api import Registry
from codt_tools.registry.db import SCHEMA_VERSION, connect
from codt_tools.registry.versions import (
    SUPPORTED_CONVENTIONS,
    IncompatibleConventionsError,
    check_conventions,
    sha256_file,
)

__all__ = [
    "Registry",
    "SCHEMA_VERSION",
    "SUPPORTED_CONVENTIONS",
    "IncompatibleConventionsError",
    "check_conventions",
    "connect",
    "sha256_file",
]
