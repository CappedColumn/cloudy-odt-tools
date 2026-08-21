"""SLURM and build-architecture introspection for CODT job submission.

CODT ships architecture-tuned builds (e.g. ``-march=znver2`` for the AMD
Rome nodes on notchpeak). An arch-tuned binary is **not portable**: running
a znver2 build on an Intel Skylake node raises SIGILL. Scheduling therefore
has to know what the binary was built for.

That information is readable straight off the executable. CODT's
``fpm.toml`` link-time flags embed an ``-Wl,-rpath=`` pointing at the spack
netCDF build it linked against, and CHPC's spack tree is named by target
microarchitecture (``linux-rocky8-zen2``, ``linux-rocky8-nehalem``, ...), so
the RPATH names the architecture::

    $ readelf -d build/.../app/CODT | grep -oE 'linux-rocky8-[a-z0-9_]+'
    linux-rocky8-zen2

This reads *which netCDF build was linked*, not CODT's own ``-march``. It is
a reliable proxy only because CODT's fpm profiles pair an ``optimized-<tag>``
feature with its matching ``netcdf-<tag>`` atomically -- a documented
invariant in CODT's ``fpm.toml.template``. A hand-mixed profile would make
the signal lie.

Every function here degrades gracefully when the underlying tool is missing
or we are off-cluster, returning ``None`` or an empty result rather than
raising, so tests and non-CHPC use keep working. All parsing functions accept
injectable text so they can be tested without a scheduler.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Iterable, Union

__all__ = [
    "ARCH_CONSTRAINT",
    "detect_build_arch",
    "parse_build_arch",
    "partition_cluster",
    "partition_nodes",
    "cores_for_constraint",
    "user_slurm_targets",
    "validate_target",
    "render_constraint",
    "parse_partition_cluster",
    "parse_partition_nodes",
    "parse_user_slurm_targets",
    "parse_sacctmgr_targets",
]

# Spack architecture token -> SLURM node feature(s) that can run that code.
#
# This table is the one place to update when CODT adds an architecture-tuned
# build profile; it mirrors the (compiler, arch) table in CODT's ``fpm_env``.
# Values are the node *features* reported by ``sinfo %f`` on CHPC.
#
# ISA-superset relations are deliberately NOT encoded (znver2 code does run
# on znver4 hardware): exact match is the safe default, and broadening is an
# explicit user override via the ``constraint`` argument.
ARCH_CONSTRAINT: dict[str, tuple[str, ...] | None] = {
    # Portable baselines -- run anywhere, no constraint needed.
    "nehalem": None,
    "x86_64": None,
    "westmere": None,
    "sandybridge": None,
    # Intel
    "skylake": ("skl", "csl"),          # Cascade Lake runs Skylake code
    "skylake_avx512": ("skl", "csl"),
    # AMD
    "zen2": ("rom",),                   # notchpeak Rome
    "zen4": ("gen",),                   # granite Genoa
}

_ARCH_RE = re.compile(r"linux-rocky8-([a-z0-9_]+)")

# "PREEMPTABLE CPU --partition=granite --qos=granite-freecycle --account=krueger 58%"
_MYCHPC_RE = re.compile(
    r"--partition=(?P<partition>\S+)\s+"
    r"--qos=(?P<qos>\S+)\s+"
    r"--account=(?P<account>\S+)"
)

_SUBPROCESS_TIMEOUT = 30

# Node features that describe core count (c64), memory (m256), interconnect
# (cx5) or site (chpc) rather than CPU type. Filtered out of error messages so
# the CPU-type features stay readable.
_NOISE_FEATURE_RE = re.compile(r"^(c|m|cx)\d+$")


def _cpu_features(features: Iterable[str]) -> list[str]:
    """Node features likely to describe CPU type, for error messages."""
    return sorted(
        f for f in features
        if f != "chpc" and not _NOISE_FEATURE_RE.match(f)
    )


# ----------------------------------------------------------------------
# Build architecture detection
# ----------------------------------------------------------------------

def parse_build_arch(readelf_output: str) -> str | None:
    """Extract the spack architecture token from ``readelf -d`` output.

    Parameters
    ----------
    readelf_output : str
        Text of ``readelf -d <executable>``.

    Returns
    -------
    str or None
        The architecture token (e.g. ``"zen2"``), or ``None`` if no spack
        path is present (a non-CHPC-linked binary) or if more than one
        distinct token is found (ambiguous -- a warning is issued).

    Notes
    -----
    ``None`` means *unknown*, not *portable*: callers must not assume an
    unconstrained binary is safe to run anywhere.
    """
    tokens = set(_ARCH_RE.findall(readelf_output))
    if not tokens:
        return None
    if len(tokens) > 1:
        warnings.warn(
            "Executable links spack libraries from more than one "
            f"architecture tree ({', '.join(sorted(tokens))}); cannot "
            "determine build architecture. No --constraint will be applied.",
            stacklevel=2,
        )
        return None
    return tokens.pop()


def detect_build_arch(executable: Union[str, Path]) -> str | None:
    """Detect the target microarchitecture of a CODT binary.

    Runs ``readelf -d`` on the executable and reads the spack architecture
    token out of the embedded netCDF RPATH.

    Parameters
    ----------
    executable : str or Path
        Path to the CODT binary.

    Returns
    -------
    str or None
        Architecture token (e.g. ``"zen2"``), or ``None`` when the binary
        does not exist, ``readelf`` is unavailable, or the architecture
        cannot be determined unambiguously.
    """
    exe = Path(executable).expanduser()
    if not exe.is_file():
        return None
    if shutil.which("readelf") is None:
        return None
    try:
        proc = subprocess.run(
            ["readelf", "-d", str(exe)],
            capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return parse_build_arch(proc.stdout)


def render_constraint(features: Iterable[str] | None) -> str | None:
    """Render node features as a SLURM ``--constraint`` expression.

    A single feature renders bare (``"rom"``); several render as an OR
    expression (``"skl|csl"``). Returns ``None`` for an empty input.
    """
    if not features:
        return None
    feats = list(features)
    return feats[0] if len(feats) == 1 else "|".join(feats)


# ----------------------------------------------------------------------
# Scheduler introspection
# ----------------------------------------------------------------------

def _run(cmd: list[str]) -> str | None:
    """Run a command, returning stdout, or None if it is unusable."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def parse_partition_cluster(sinfo_output: str, partition: str) -> str | None:
    """Map a partition to its cluster from ``sinfo -M all`` output.

    ``sinfo -M all`` groups its output under ``CLUSTER: <name>`` headers;
    this walks those sections looking for the partition. Partition names
    carry a trailing ``*`` when they are the cluster default.
    """
    want = partition.rstrip("*")
    cluster: str | None = None
    for line in sinfo_output.splitlines():
        stripped = line.strip()
        if stripped.startswith("CLUSTER:"):
            cluster = stripped.split(":", 1)[1].strip()
            continue
        if not stripped or cluster is None:
            continue
        name = stripped.split()[0].rstrip("*")
        if name == want:
            return cluster
    return None


def partition_cluster(partition: str) -> str | None:
    """Return the cluster a partition lives on, or None if not found.

    Needed because submitting to a partition on another cluster requires
    ``sbatch -M <cluster>``.
    """
    out = _run(["sinfo", "-M", "all", "-o", "%P"])
    if out is None:
        return None
    return parse_partition_cluster(out, partition)


def parse_partition_nodes(sinfo_output: str) -> list[tuple[set[str], int]]:
    """Parse ``sinfo -h -o "%f|%c"`` output into (features, cores) pairs.

    Each line describes one node group, e.g.
    ``chpc,rom,c64,m256,cx5|64``. Unparseable lines are skipped.
    """
    groups: list[tuple[set[str], int]] = []
    for line in sinfo_output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        feature_text, core_text = parts[0], parts[1]
        try:
            cores = int(core_text.strip())
        except ValueError:
            continue
        features = {
            f.strip() for f in feature_text.split(",")
            if f.strip() and f.strip() != "(null)"
        }
        groups.append((features, cores))
    return groups


def partition_nodes(
    partition: str, cluster: str | None = None
) -> list[tuple[set[str], int]]:
    """Return (feature set, core count) for each node group in a partition.

    Returns an empty list when ``sinfo`` is unavailable or the partition is
    unknown.
    """
    cmd = ["sinfo"]
    if cluster:
        cmd += ["-M", cluster]
    cmd += ["-p", partition, "-h", "-o", "%f|%c"]
    out = _run(cmd)
    if out is None:
        return []
    return parse_partition_nodes(out)


def cores_for_constraint(
    partition: str,
    cluster: str | None = None,
    features: Iterable[str] | None = None,
    nodes: list[tuple[set[str], int]] | None = None,
) -> int | None:
    """Core count to pack per node, given an optional feature constraint.

    Returns the **minimum** core count over the node groups that match any
    of ``features`` (or over the whole partition when ``features`` is
    ``None``). The minimum is deliberate: a constraint like ``skl`` spans
    32- and 36-core nodes, and packing to 36 would oversubscribe the 32-core
    ones.

    Parameters
    ----------
    partition, cluster : str
        Partition and cluster to query (ignored when ``nodes`` is given).
    features : iterable of str, optional
        Node features the job will be constrained to.
    nodes : list of (set, int), optional
        Pre-parsed node groups, injected for testing.

    Returns
    -------
    int or None
        Core count, or ``None`` when no node group matches or introspection
        is unavailable.
    """
    if nodes is None:
        nodes = partition_nodes(partition, cluster)
    if not nodes:
        return None
    wanted = set(features) if features else None
    matching = [
        cores for feats, cores in nodes
        if wanted is None or (feats & wanted)
    ]
    if not matching:
        return None
    return min(matching)


def parse_user_slurm_targets(mychpc_output: str) -> list[dict[str, str]]:
    """Parse ``mychpc batch`` output into partition/qos/account triples.

    Note that partition and qos are frequently *different* (partition
    ``granite`` uses qos ``granite-freecycle``), so all three are carried.
    """
    targets: list[dict[str, str]] = []
    for line in mychpc_output.splitlines():
        match = _MYCHPC_RE.search(line)
        if match:
            targets.append(match.groupdict())
    return targets


def parse_sacctmgr_targets(sacctmgr_output: str) -> list[dict[str, str]]:
    """Parse ``sacctmgr -n -P show assoc ...`` as a fallback for mychpc.

    Expects ``format=cluster,account,partition,qos``. Associations with an
    empty partition field apply to every partition on the cluster and are
    skipped, since they cannot be matched against a specific target.
    """
    targets: list[dict[str, str]] = []
    for line in sacctmgr_output.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 4:
            continue
        account, partition, qos = fields[1], fields[2], fields[3]
        if not partition:
            continue
        for single_qos in (qos.split(",") if qos else [""]):
            targets.append({
                "partition": partition,
                "qos": single_qos,
                "account": account,
            })
    return targets


def user_slurm_targets() -> list[dict[str, str]]:
    """Return the partition/qos/account triples this user may submit to.

    Prefers ``mychpc batch`` (the authoritative, copy-verbatim source per
    CHPC's own guidance), falling back to ``sacctmgr``. Returns an empty
    list when neither is available -- callers must treat that as "unknown",
    not "not entitled".
    """
    out = _run(["mychpc", "batch"])
    if out:
        targets = parse_user_slurm_targets(out)
        if targets:
            return targets
    user = os.environ.get("USER", "")
    if not user:
        return []
    out = _run([
        "sacctmgr", "-n", "-P", "show", "assoc", f"user={user}",
        "format=cluster,account,partition,qos",
    ])
    if out is None:
        return []
    return parse_sacctmgr_targets(out)


# ----------------------------------------------------------------------
# Preflight validation
# ----------------------------------------------------------------------

def validate_target(
    partition: str,
    account: str,
    qos: str | None = None,
    features: Iterable[str] | None = None,
    cluster: str | None = None,
    build_arch: str | None = None,
    targets: list[dict[str, str]] | None = None,
    nodes: list[tuple[set[str], int]] | None = None,
) -> list[str]:
    """Check a submission target, returning a list of human-readable problems.

    An empty list means the target looks good. Three checks run:

    1. **Entitlement** -- the (partition, account, qos) triple appears in
       :func:`user_slurm_targets`.
    2. **Schedulability** -- if ``features`` are required, at least one node
       in the partition carries one of them. This is the check that catches
       a zen2 binary aimed at ``notchpeak-shared-short`` (no Rome nodes),
       which would otherwise sit unschedulable forever.
    3. **Silent-SIGILL risk** -- the partition mixes matching and
       non-matching nodes but no constraint is being applied.

    Introspection that is unavailable (off-cluster, no ``sinfo``) yields no
    problems rather than false failures.

    Parameters
    ----------
    targets, nodes : optional
        Pre-fetched entitlement triples and node groups, injected for
        testing. Queried live when omitted.

    Returns
    -------
    list of str
        Problem descriptions, most specific first.
    """
    problems: list[str] = []
    feats = set(features) if features else set()

    if targets is None:
        targets = user_slurm_targets()
    if targets:
        def matches(t: dict[str, str]) -> bool:
            if t.get("partition") != partition or t.get("account") != account:
                return False
            return qos is None or t.get("qos") == qos

        if not any(matches(t) for t in targets):
            offered = sorted({
                f"--partition={t['partition']} --qos={t['qos']} "
                f"--account={t['account']}"
                for t in targets
            })
            problems.append(
                f"Not entitled to partition={partition} account={account}"
                + (f" qos={qos}" if qos else "")
                + ". Valid targets for this user:\n  "
                + "\n  ".join(offered)
            )

    if nodes is None:
        nodes = partition_nodes(partition, cluster)

    if nodes:
        available = _cpu_features({f for group, _ in nodes for f in group})
        if feats:
            if not any(group & feats for group, _ in nodes):
                arch_note = (
                    f"Executable is built for '{build_arch}' "
                    if build_arch else ""
                )
                problems.append(
                    f"{arch_note}(needs node feature "
                    f"{' or '.join(sorted(feats))}), but partition "
                    f"'{partition}' has no such nodes (available features: "
                    f"{', '.join(available)}). The job would be "
                    f"unschedulable, or crash with SIGILL. Choose a "
                    f"partition with matching nodes, or rebuild CODT with "
                    f"the baseline 'release' profile."
                )
        elif build_arch is None:
            distinct = {
                frozenset(group & {
                    "rom", "skl", "csl", "gen", "npl", "mil", "emr",
                }) for group, _ in nodes
            }
            if len(distinct) > 1:
                problems.append(
                    f"Build architecture of the executable is unknown and "
                    f"partition '{partition}' mixes CPU types "
                    f"({', '.join(available)}). If the binary is "
                    f"architecture-tuned it will crash on some nodes. Pass "
                    f"an explicit constraint= to pin it."
                )

    return problems
