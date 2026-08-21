"""Tests for case validation — the mirrors of CODT's own startup aborts."""

from __future__ import annotations

import pytest

from codt_tools.case import Case, lem_turbulence_scales


# ======================================================================
# Derived LEM turbulence scales (mirror of CODT src/LEM.f90)
# ======================================================================


class TestLEMTurbulenceScales:
    """Tests for lem_turbulence_scales()."""

    def test_grid_limited_regime(self) -> None:
        # eps = 0.01, H = 1, N = 1025: 6*dz just exceeds the diffusivity
        # length scale, so the floor is exactly 6 cells and f ~ 1.
        s = lem_turbulence_scales(1025, 1.0, 0.01)
        assert s["smallest_eddy_gridpoints"] == 6
        assert s["diffusivity_enhancement"] == pytest.approx(1.001, abs=1e-3)

    def test_diffusivity_limited_regime_overshoots(self) -> None:
        # N = 1045 puts 6*dz just below the handoff, so the 3-cell quantum
        # rounds up to 9 cells and f jumps -- f is a step function of the
        # grid, not ~1.
        s = lem_turbulence_scales(1045, 1.0, 0.01)
        assert s["smallest_eddy_gridpoints"] == 9
        assert s["diffusivity_enhancement"] == pytest.approx(1.675, abs=1e-3)

    def test_enhancement_never_below_one(self) -> None:
        for n in range(200, 3000, 37):
            s = lem_turbulence_scales(n, 1.0, 0.01)
            assert s["diffusivity_enhancement"] >= 1.0

    def test_gridpoints_consistent_with_scale(self) -> None:
        n, h = 2000, 1.0
        s = lem_turbulence_scales(n, h, 0.01)
        assert s["smallest_eddy_scale"] == pytest.approx(
            s["smallest_eddy_gridpoints"] * h / n
        )
        assert s["smallest_eddy_gridpoints"] % 3 == 0

    def test_matches_codt_reference_values(self) -> None:
        # Pinned against the LEM.* global attributes of a real CODT v3.0.0
        # (e1c03be) parcel run at N=2000, H=1, eps=0.01. Guards the
        # single-precision `1./3.` / `4./3.` literals: dropping them shifts
        # diffusivity_length_scale by ~4e-8 relative, which this catches.
        s = lem_turbulence_scales(2000, 1.0, 0.01)
        assert s["diffusivity_length_scale"] == pytest.approx(
            0.005849128279333603, rel=1e-7
        )
        assert s["actual_kolmogorov_scale"] == pytest.approx(
            0.00075762133825086, rel=1e-7
        )
        assert s["diffusivity_enhancement"] == pytest.approx(
            1.0345388541686187, rel=1e-7
        )
        assert s["smallest_eddy_gridpoints"] == 12
        assert s["smallest_eddy_scale"] == pytest.approx(0.006)

    def test_kolmogorov_scale_is_diagnostic_only(self) -> None:
        # (nu**3/eps)**0.25 is always ~8x below the diffusivity handoff, so
        # it can never win the max that sets the smallest eddy.
        s = lem_turbulence_scales(2000, 1.0, 0.01)
        assert s["actual_kolmogorov_scale"] < s["diffusivity_length_scale"]
        assert s["actual_kolmogorov_scale"] < s["smallest_eddy_scale"]



# ======================================================================
# Validate
# ======================================================================


class TestCaseValidate:
    """Tests for validate()."""

    def test_valid_chamber_passes(self) -> None:
        cfg = Case()
        cfg.validate()  # should not raise

    def test_valid_parcel_passes(self) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel")
        cfg.validate()

    def test_chamber_tdiff_zero_fails(self) -> None:
        cfg = Case()
        cfg.set(tdiff=0.0)
        with pytest.raises(ValueError, match="tdiff"):
            cfg.validate()

    def test_parcel_without_legs_fails(self) -> None:
        """Parcel mode is defined by having a trajectory, not by a filename."""
        cfg = Case()
        cfg.set(simulation_mode="parcel")
        cfg.parcel.set(segment_coord=[], velocity=[])
        with pytest.raises(ValueError, match="at least one leg"):
            cfg.validate()

    def test_parcel_bad_rh_fails(self) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel", initial_rh=1.5)
        with pytest.raises(ValueError, match="initial_rh"):
            cfg.validate()

    def test_parcel_entrainment_needs_env(self) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                do_entrainment=True)
        with pytest.raises(ValueError, match="environmental sounding"):
            cfg.validate()

    def test_parcel_entrainment_with_env_passes(self) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                do_entrainment=True)
        cfg.parcel.set_env_profile(
            height=[0.0, 2000.0],
            pressure=[100000.0, 80000.0],
            temperature=[300.0, 285.0],
            RH=[0.8, 0.5],
        )
        cfg.validate()

    def _entraining_parcel(self) -> Case:
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                do_entrainment=True)
        cfg.parcel.set_env_profile(
            height=[0.0, 2000.0],
            pressure=[100000.0, 80000.0],
            temperature=[300.0, 285.0],
            RH=[0.8, 0.5],
        )
        return cfg

    def test_parcel_entrainment_chunk_needs_a_gridcell(self) -> None:
        # int(0.001 * 1024) = 1 cell cannot be split into 2 chunks.
        cfg = self._entraining_parcel()
        cfg.set(n=1024, psigma=0.001, n_blob=2)
        with pytest.raises(ValueError, match=r"int\(psigma \* N\)"):
            cfg.validate()

    def test_parcel_entrainment_psigma_times_n_blob_over_one_passes(
        self,
    ) -> None:
        # Rejected before CODT 3.1.0: psigma is now the total fraction
        # replaced, so n_blob no longer multiplies it.
        cfg = self._entraining_parcel()
        cfg.set(psigma=0.5, n_blob=3)
        cfg.validate()

    def test_parcel_entrainment_bad_psigma_fails(self) -> None:
        cfg = self._entraining_parcel()
        cfg.set(psigma=1.0)
        with pytest.raises(ValueError, match="psigma must be in"):
            cfg.validate()

    def test_parcel_entrainment_leg_schedule_checked(self) -> None:
        cfg = self._entraining_parcel()
        cfg.set(n=1024)
        cfg.parcel.set_entrainment_schedule(
            ent_rate=[1.0], n_blob=[5], psigma=[0.001]
        )
        with pytest.raises(ValueError, match=r"int\(psigma \* N\)"):
            cfg.validate()

    def test_parcel_environment_mode_needs_env(self) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                pressure_mode="environment")
        with pytest.raises(ValueError, match="environmental sounding"):
            cfg.validate()

    def test_parcel_bad_vertical_axis_fails(self) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                vertical_axis="altitude")
        with pytest.raises(ValueError, match="vertical_axis"):
            cfg.validate()

    def test_parcel_bad_pressure_mode_fails(self) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                pressure_mode="isobaric")
        with pytest.raises(ValueError, match="pressure_mode"):
            cfg.validate()

    def test_parcel_leg_pointing_away_fails(self) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel")
        cfg.parcel.set(segment_coord=[1000.0], velocity=[-1.0])
        with pytest.raises(ValueError, match="points away from its target"):
            cfg.validate()

    def test_parcel_legs_validate_against_initial_height(self) -> None:
        # Launching at 1200 m makes a 1000 m target a descent, so the
        # ascending velocity that would pass from the ground now fails.
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                initial_height=1200.0)
        cfg.parcel.set(segment_coord=[1000.0], velocity=[1.0])
        with pytest.raises(ValueError, match="points away from its target"):
            cfg.validate()

    # -- Derived LEM scales (CODT 3.0.0) --

    def test_parcel_domain_too_small_for_smallest_eddy_fails(self) -> None:
        # N = 4 cannot hold the 6-cell structural floor.
        cfg = Case()
        cfg.set(simulation_mode="parcel", n=4, h=1.0)
        with pytest.raises(ValueError, match="cannot contain the smallest eddy"):
            cfg.validate()

    def test_parcel_no_inertial_range_fails(self) -> None:
        # A coarse grid pushes 6*dz above integral_length_scale.
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                n=100, h=1.0, integral_length_scale=0.01)
        with pytest.raises(ValueError, match="No inertial range"):
            cfg.validate()

    def test_parcel_narrow_inertial_range_warns(self) -> None:
        # The defaults themselves sit here (l_small = 6 mm vs L = 10 mm),
        # matching the warning CODT emits on its bundled params.nml.
        cfg = Case()
        cfg.set(simulation_mode="parcel")
        with pytest.warns(UserWarning, match="inertial range is nearly absent"):
            cfg.validate()

    def test_lem_scales_not_checked_without_turbulence(self) -> None:
        cfg = Case()
        cfg.set(simulation_mode="parcel",
                n=4, h=1.0, do_turbulence=False)
        cfg.validate()  # should not raise

    def test_chamber_lmin_below_floor_fails(self) -> None:
        cfg = Case()
        cfg.set(lmin=3)
        with pytest.raises(ValueError, match="smallest representable eddy"):
            cfg.validate()

    def test_chamber_lmin_not_multiple_of_three_fails(self) -> None:
        cfg = Case()
        cfg.set(lmin=8)
        with pytest.raises(ValueError, match="multiple of 3"):
            cfg.validate()

    def test_chamber_lmin_valid_passes(self) -> None:
        cfg = Case()
        cfg.set(lmin=9)
        cfg.validate()

    def test_bad_trajectory_window(self) -> None:
        cfg = Case()
        cfg.set(
            write_trajectories=True,
            trajectory_start=100.0,
            trajectory_end=50.0,
            tmax=3600.0,
        )
        with pytest.raises(ValueError, match="trajectory_start"):
            cfg.validate()

    def test_trajectory_exceeds_tmax(self) -> None:
        cfg = Case()
        cfg.set(
            write_trajectories=True,
            trajectory_start=100.0,
            trajectory_end=5000.0,
            tmax=3600.0,
        )
        with pytest.raises(ValueError, match="trajectory_end"):
            cfg.validate()


