"""Tests for the simulation layer's boundaries.

Three properties, none of them about analysis maths:

- ``Simulation.from_run`` is the bridge from the run layer, and it refuses an
  unfinished run rather than reporting on partial output.
- Output written in an unknown format **warns and still opens**. The data
  already exists and a scientist must be able to look at it; input files
  raise instead, because a wrong-format input silently produces a wrong
  simulation.
- Reading your own output does not require the registry. That edge
  (``simulation.py`` importing ``registry.versions``) is what the layering
  exists to remove, so it is pinned here.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import netCDF4 as nc
import pytest

from codt_tools.case import Case
from codt_tools.run import Run
from codt_tools.simulation import OUTPUT_CONVENTIONS, Simulation


@contextmanager
def no_conventions_warning():
    """Assert that nothing warns about conventions inside the block."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    offending = [w for w in caught if "conventions" in str(w.message)]
    assert not offending, f"unexpected warning: {offending}"


@pytest.fixture
def case() -> Case:
    """A case whose name matches the fixture output files."""
    cfg = Case()
    cfg.set(simulation_name="test_sim")
    return cfg


class TestFromRun:
    """The bridge from a Run to its output."""

    def test_opens_a_completed_run(
        self, case: Case, completed_run_dir: Path
    ) -> None:
        run = Run(case, completed_run_dir / "CODT", completed_run_dir)

        sim = Simulation.from_run(run)

        assert sim.T.shape[0] > 0

    def test_refuses_an_incomplete_run(
        self, case: Case, completed_run_dir: Path
    ) -> None:
        run = Run(case, completed_run_dir / "CODT", completed_run_dir)
        run.done_marker.unlink()

        with pytest.raises(FileNotFoundError, match="not complete"):
            Simulation.from_run(run)

    def test_open_simulation_delegates_here(
        self, case: Case, completed_run_dir: Path
    ) -> None:
        """Run.open_simulation and Simulation.from_run are one path, so they
        cannot drift apart."""
        run = Run(case, completed_run_dir / "CODT", completed_run_dir)

        assert type(run.open_simulation()) is type(Simulation.from_run(run))


class TestConventions:
    """One format string, and a mismatch is a warning."""

    def test_matching_conventions_are_quiet(
        self, completed_run_dir: Path
    ) -> None:
        with no_conventions_warning():
            Simulation(completed_run_dir / "output")

    def test_mismatch_warns_but_still_opens(
        self, completed_run_dir: Path
    ) -> None:
        output = completed_run_dir / "output"
        with nc.Dataset(output / "test_sim.nc", "a") as ds:
            ds.setncattr("conventions", "CODT_output_v99")

        with pytest.warns(UserWarning, match="CODT_output_v99"):
            sim = Simulation(output)

        # The point of warning rather than raising: the data is still there.
        assert sim.T.shape[0] > 0

    def test_missing_conventions_warns(self, completed_run_dir: Path) -> None:
        output = completed_run_dir / "output"
        with nc.Dataset(output / "test_sim.nc", "a") as ds:
            ds.delncattr("conventions")

        with pytest.warns(UserWarning, match=OUTPUT_CONVENTIONS):
            Simulation(output)


class TestNoRegistryDependency:
    """Analysis must not drag in the bookkeeping layer."""

    def test_importing_the_simulation_layer_leaves_registry_alone(
        self,
    ) -> None:
        """Run in a fresh interpreter: this test process has almost certainly
        imported the registry already, via other tests."""
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import codt_tools.simulation, sys;"
                "leaked = [m for m in sys.modules"
                " if m.startswith('codt_tools.registry')];"
                "print(leaked)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert proc.stdout.strip() == "[]", (
            f"codt_tools.simulation pulled in the registry: {proc.stdout}"
        )

    def test_top_level_import_leaves_registry_alone(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import codt_tools, sys;"
                "leaked = [m for m in sys.modules"
                " if m.startswith('codt_tools.registry')];"
                "print(leaked)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert proc.stdout.strip() == "[]", (
            f"importing codt_tools pulled in the registry: {proc.stdout}"
        )
