"""Tests for SLURM and build-architecture introspection.

Every parsing function takes injectable text, so nothing here needs a
scheduler or a CHPC binary.
"""

from __future__ import annotations

import pytest

from codt_tools.slurm import (
    ARCH_CONSTRAINT,
    cores_for_constraint,
    detect_build_arch,
    parse_build_arch,
    parse_partition_cluster,
    parse_partition_nodes,
    parse_sacctmgr_targets,
    parse_user_slurm_targets,
    render_constraint,
    validate_target,
)

# Abridged from a real `readelf -d` on a notchpeak-rome CODT build.
READELF_ZEN2 = """
Dynamic section at offset 0x1a2b28 contains 30 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)  Shared library: [libnetcdff.so.7]
 0x000000000000000f (RPATH)   Library rpath: [/uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-zen2/gcc-11.2.0/netcdf-fortran-4.5.4-abc/lib:/uufs/chpc.utah.edu/sys/spack/v019/linux-rocky8-zen2/gcc-11.2.0/netcdf-c-4.8.1-def/lib]
 0x000000000000000c (INIT)    0x4021c8
"""

SINFO_FREECYCLE = """chpc,rom,c64,m256,cx5|64
chpc,skl,c32,m192,cx4|32
chpc,skl,c36,m768,cx4|36
chpc,csl,c40,m192,cx5|40
"""

SINFO_SHARED_SHORT = """chpc,csl,t4,c52,m384,cx6|52
chpc,npl,1080ti,c64,m512,cx5|64
"""

MYCHPC_BATCH = """GENERAL CPU --partition=kingspeak --qos=kingspeak --account=krueger 22%
PREEMPTABLE CPU --partition=granite --qos=granite-freecycle --account=krueger 58%
PREEMPTABLE CPU --partition=notchpeak-freecycle --qos=notchpeak-freecycle --account=krueger 24%
"""

SINFO_CLUSTERS = """CLUSTER: granite
PARTITION
granite*
granite-gpu

CLUSTER: notchpeak
PARTITION
notchpeak-freecycle
notchpeak-shared-short
"""


class TestParseBuildArch:
    def test_single_token(self) -> None:
        assert parse_build_arch(READELF_ZEN2) == "zen2"

    def test_no_spack_path_returns_none(self) -> None:
        """A locally built, non-CHPC-linked binary yields unknown."""
        assert parse_build_arch("RPATH: [/usr/local/lib]") is None

    def test_multiple_tokens_warns_and_returns_none(self) -> None:
        """Ambiguous is treated as unknown, never as 'pick the first'."""
        text = "linux-rocky8-zen2/a and linux-rocky8-skylake/b"
        with pytest.warns(UserWarning, match="more than one"):
            assert parse_build_arch(text) is None

    def test_missing_executable(self, tmp_path) -> None:
        assert detect_build_arch(tmp_path / "nope") is None


class TestArchConstraintTable:
    @pytest.mark.parametrize(
        "arch,expected",
        [
            ("zen2", ("rom",)),
            ("zen4", ("gen",)),
            ("skylake", ("skl", "csl")),
            ("nehalem", None),
        ],
    )
    def test_lookups(self, arch, expected) -> None:
        assert ARCH_CONSTRAINT[arch] == expected

    def test_unknown_arch_absent(self) -> None:
        assert ARCH_CONSTRAINT.get("some_future_arch") is None

    @pytest.mark.parametrize(
        "features,expected",
        [(("rom",), "rom"), (("skl", "csl"), "skl|csl"), (None, None),
         ((), None)],
    )
    def test_render_constraint(self, features, expected) -> None:
        assert render_constraint(features) == expected


class TestParsePartitionNodes:
    def test_parses_features_and_cores(self) -> None:
        nodes = parse_partition_nodes(SINFO_FREECYCLE)
        assert len(nodes) == 4
        features, cores = nodes[0]
        assert "rom" in features and cores == 64

    def test_skips_malformed_lines(self) -> None:
        assert parse_partition_nodes("garbage\n\nchpc,rom|64\n") == [
            ({"chpc", "rom"}, 64)
        ]

    def test_handles_null_features(self) -> None:
        nodes = parse_partition_nodes("(null)|16\n")
        assert nodes == [(set(), 16)]


class TestCoresForConstraint:
    def test_min_over_matching_nodes(self) -> None:
        """skl spans 32- and 36-core nodes; packing must use 32."""
        nodes = parse_partition_nodes(SINFO_FREECYCLE)
        assert cores_for_constraint("p", None, ("skl",), nodes=nodes) == 32

    def test_single_matching_group(self) -> None:
        nodes = parse_partition_nodes(SINFO_FREECYCLE)
        assert cores_for_constraint("p", None, ("rom",), nodes=nodes) == 64

    def test_no_constraint_uses_whole_partition(self) -> None:
        nodes = parse_partition_nodes(SINFO_FREECYCLE)
        assert cores_for_constraint("p", None, None, nodes=nodes) == 32

    def test_no_match_returns_none(self) -> None:
        nodes = parse_partition_nodes(SINFO_SHARED_SHORT)
        assert cores_for_constraint("p", None, ("rom",), nodes=nodes) is None

    def test_empty_nodes_returns_none(self) -> None:
        assert cores_for_constraint("p", None, ("rom",), nodes=[]) is None


class TestUserSlurmTargets:
    def test_parses_triples(self) -> None:
        targets = parse_user_slurm_targets(MYCHPC_BATCH)
        assert len(targets) == 3
        assert {
            "partition": "granite",
            "qos": "granite-freecycle",
            "account": "krueger",
        } in targets

    def test_partition_and_qos_can_differ(self) -> None:
        """granite uses qos granite-freecycle: all three must be carried."""
        granite = next(
            t for t in parse_user_slurm_targets(MYCHPC_BATCH)
            if t["partition"] == "granite"
        )
        assert granite["qos"] != granite["partition"]

    def test_sacctmgr_fallback_expands_qos_list(self) -> None:
        text = "notchpeak|krueger|notchpeak-freecycle|qos1,qos2\n"
        assert parse_sacctmgr_targets(text) == [
            {"partition": "notchpeak-freecycle", "qos": "qos1",
             "account": "krueger"},
            {"partition": "notchpeak-freecycle", "qos": "qos2",
             "account": "krueger"},
        ]

    def test_sacctmgr_skips_partitionless_assoc(self) -> None:
        assert parse_sacctmgr_targets("notchpeak|krueger||qos\n") == []


class TestParsePartitionCluster:
    @pytest.mark.parametrize(
        "partition,cluster",
        [
            ("notchpeak-freecycle", "notchpeak"),
            ("notchpeak-shared-short", "notchpeak"),
            ("granite", "granite"),      # default partition, trailing '*'
            ("granite-gpu", "granite"),
        ],
    )
    def test_maps_partition_to_cluster(self, partition, cluster) -> None:
        assert parse_partition_cluster(SINFO_CLUSTERS, partition) == cluster

    def test_unknown_partition(self) -> None:
        assert parse_partition_cluster(SINFO_CLUSTERS, "nope") is None


class TestValidateTarget:
    TARGETS = [
        {"partition": "notchpeak-freecycle", "qos": "notchpeak-freecycle",
         "account": "krueger"},
    ]

    def test_entitled_and_schedulable_passes(self) -> None:
        nodes = parse_partition_nodes(SINFO_FREECYCLE)
        assert validate_target(
            "notchpeak-freecycle", "krueger", "notchpeak-freecycle",
            features=("rom",), build_arch="zen2",
            targets=self.TARGETS, nodes=nodes,
        ) == []

    def test_not_entitled(self) -> None:
        nodes = parse_partition_nodes(SINFO_FREECYCLE)
        problems = validate_target(
            "notchpeak", "krueger", "notchpeak",
            features=("rom",), targets=self.TARGETS, nodes=nodes,
        )
        assert len(problems) == 1
        assert "Not entitled" in problems[0]
        # The message must name a usable alternative.
        assert "notchpeak-freecycle" in problems[0]

    def test_no_matching_feature_is_caught(self) -> None:
        """The zen2-binary -> notchpeak-shared-short case."""
        nodes = parse_partition_nodes(SINFO_SHARED_SHORT)
        problems = validate_target(
            "notchpeak-shared-short", "krueger", "notchpeak-shared-short",
            features=("rom",), build_arch="zen2",
            targets=[{"partition": "notchpeak-shared-short",
                      "qos": "notchpeak-shared-short",
                      "account": "krueger"}],
            nodes=nodes,
        )
        assert len(problems) == 1
        assert "zen2" in problems[0] and "rom" in problems[0]
        assert "SIGILL" in problems[0]
        # Core-count/memory features are noise in this message.
        assert "c52" not in problems[0] and "m384" not in problems[0]

    def test_unknown_arch_on_mixed_partition_warns(self) -> None:
        nodes = parse_partition_nodes(SINFO_FREECYCLE)
        problems = validate_target(
            "notchpeak-freecycle", "krueger", "notchpeak-freecycle",
            features=None, build_arch=None,
            targets=self.TARGETS, nodes=nodes,
        )
        assert len(problems) == 1
        assert "mixes CPU types" in problems[0]

    def test_uniform_partition_with_unknown_arch_is_quiet(self) -> None:
        nodes = parse_partition_nodes("chpc,gen,c96,m768|96\n")
        assert validate_target(
            "granite", "krueger", "granite-freecycle",
            features=None, build_arch=None,
            targets=[{"partition": "granite", "qos": "granite-freecycle",
                      "account": "krueger"}],
            nodes=nodes,
        ) == []

    def test_qos_none_matches_any_qos(self) -> None:
        nodes = parse_partition_nodes(SINFO_FREECYCLE)
        assert validate_target(
            "notchpeak-freecycle", "krueger", None,
            features=("rom",), targets=self.TARGETS, nodes=nodes,
        ) == []

    def test_no_introspection_yields_no_false_failures(self) -> None:
        """Off-cluster: unknown must not read as 'not entitled'."""
        assert validate_target(
            "whatever", "acct", "qos", features=("rom",),
            targets=[], nodes=[],
        ) == []
