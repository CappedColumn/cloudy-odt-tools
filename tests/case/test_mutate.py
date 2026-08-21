"""Tests for points, designs, and the paths that address a case.

The shapes exercised here are the ones real ensembles have: axes whose values
vary together (LHS samples, paired parameters), whole-component swaps, derived
values, control groups appended to a sweep, and filtered combinations.
"""

from __future__ import annotations

import numpy as np
import pytest

from codt_tools.case import Aerosol, Case, Parcel, cross
from codt_tools.case.mutate import apply_point, default_name


class TestPaths:
    """One point, applied to one case."""

    def test_params_field(self) -> None:
        case = Case()
        case.apply({"params.tref": 22.0, "params.tmax": 60.0})
        assert case.params.get("tref") == 22.0
        assert case.params.get("tmax") == 60.0

    def test_params_routes_to_the_right_group(self) -> None:
        """Paths name the component, not the namelist group."""
        case = Case()
        case.apply({"params.tdiff": 12.0, "params.psigma": 0.2})
        assert case.params._data["turbulence_odt"]["tdiff"] == 12.0
        assert case.params._data["entrainment"]["psigma"] == 0.2

    def test_aerosol_field(self) -> None:
        case = Case()
        case.apply({"aerosol.aerosol_name": "KCl"})
        assert case.aerosol.aerosol_name == "KCl"

    def test_parcel_field(self) -> None:
        case = Case()
        case.apply({"parcel.velocity": [2.0]})
        np.testing.assert_allclose(case.parcel.velocity, [2.0])

    def test_whole_component_replacement(self) -> None:
        """The seed-mode axis of a real sweep assigns a prebuilt Aerosol."""
        case = Case()
        other = Aerosol()
        other.set(aerosol_name="KCl")

        case.apply({"aerosol": other})

        assert case.aerosol.aerosol_name == "KCl"

    def test_component_replacement_is_type_checked(self) -> None:
        case = Case()
        with pytest.raises(ValueError, match="Aerosol"):
            case.apply({"aerosol": Parcel()})

    def test_unprefixed_path_raises(self) -> None:
        case = Case()
        with pytest.raises(ValueError, match="no component prefix"):
            case.apply({"tref": 22.0})

    def test_unknown_component_raises(self) -> None:
        case = Case()
        with pytest.raises(ValueError, match="does not address a case"):
            case.apply({"namelist.tref": 22.0})

    def test_unknown_field_raises(self) -> None:
        case = Case()
        with pytest.raises(KeyError):
            case.apply({"params.tefr": 22.0})
        with pytest.raises(AttributeError):
            case.apply({"aerosol.not_a_field": 1.0})

    def test_staged_keys_still_refused(self) -> None:
        """write_inputs owns file locations, whichever door you come in by."""
        case = Case()
        for key in ("output_directory", "aerosol_file", "parcel_file"):
            with pytest.raises(ValueError, match="write_inputs"):
                case.apply({f"params.{key}": "/somewhere"})

    def test_apply_mutates_in_place(self) -> None:
        case = Case()
        assert case.apply({"params.tref": 22.0}) is None
        assert case.params.get("tref") == 22.0


class TestCallableValues:
    """A point can transform the current value instead of replacing it."""

    def test_transforms_a_parameter(self) -> None:
        case = Case()
        case.set(tmax=100.0)
        case.apply({"params.tmax": lambda t: 2 * t})
        assert case.params.get("tmax") == 200.0

    def test_transforms_a_component(self) -> None:
        def rename(aerosol: Aerosol) -> Aerosol:
            aerosol.set(aerosol_name="KCl")
            return aerosol

        case = Case()
        case.apply({"aerosol": rename})
        assert case.aerosol.aerosol_name == "KCl"

    def test_callable_returning_none_raises(self) -> None:
        """Modifying in place and returning nothing would blank the field."""
        case = Case()
        with pytest.raises(ValueError, match="returned None"):
            case.apply({"aerosol": lambda a: a.set(aerosol_name="KCl")})


class TestCross:
    """Building a design."""

    def test_dict_of_lists_expands_to_independent_axes(self) -> None:
        design = cross({"params.tref": [20.0, 22.0], "params.n_blob": [1, 5]})
        assert len(design) == 4
        assert design[0] == {"params.tref": 20.0, "params.n_blob": 1}

    def test_a_list_of_points_is_one_axis(self) -> None:
        """LHS samples vary together: 2 points, not 2**k combinations."""
        lhs = [
            {"params.tref": 9.0, "params.pres": 66100.0},
            {"params.tref": 11.0, "params.pres": 68100.0},
        ]
        assert len(cross(lhs)) == 2
        assert len(cross(lhs, {"params.n_blob": [1, 5]})) == 4

    def test_merges_across_axes(self) -> None:
        design = cross(
            [{"params.tref": 20.0}],
            [{"params.n_blob": 5, "params.do_entrainment": True}],
        )
        assert design == [
            {"params.tref": 20.0, "params.n_blob": 5,
             "params.do_entrainment": True}
        ]

    def test_no_axes_is_the_empty_product(self) -> None:
        assert cross() == [{}]

    def test_empty_axis_gives_no_points(self) -> None:
        assert cross([{"params.tref": 20.0}], []) == []

    def test_duplicate_path_across_axes_raises(self) -> None:
        """Crossed axes are meant to be independent; + unions designs."""
        with pytest.raises(ValueError, match="both set params.tref"):
            cross({"params.tref": [20.0]}, {"params.tref": [22.0]})

    def test_scalar_instead_of_list_raises(self) -> None:
        with pytest.raises(ValueError, match="needs a list of values"):
            cross({"params.tref": 20.0})

    def test_non_mapping_axis_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="list of points"):
            cross([20.0, 22.0])


class TestDesignsAreOrdinaryLists:
    """The point of the design being a list of dicts."""

    def test_filter_and_union(self) -> None:
        base = Case()
        base.set(simulation_name="exp")

        swept = cross(
            {"params.tref": [20.0, 22.0]},
            {"params.n_blob": [1, 5, 10]},
        )
        design = [p for p in swept if p["params.n_blob"] <= 5]
        control = [{"params.tref": 21.0, "params.do_entrainment": False}]

        cases = base.sweep(design + control)

        assert len(cases) == 5
        assert cases[-1].params.get("do_entrainment") is False

    def test_derived_values_are_built_into_the_axis(self) -> None:
        """do_entrainment follows from the sampled rate, as in SF01."""
        base = Case()
        rates = [0.0, 0.5]
        axis = [
            {"params.ent_rate": r, "params.do_entrainment": r > 0.0}
            for r in rates
        ]

        cases = base.sweep(axis)

        assert cases[0].params.get("do_entrainment") is False
        assert cases[1].params.get("do_entrainment") is True

    def test_sf01_shape(self) -> None:
        """LHS x modes x paired realizations, the shape that had to be
        hand-rolled before: 3 x 2 x 3 = 18, not 3**2 x 2 x 3**2."""
        base = Case()
        lhs = [
            {"params.ent_rate": r, "params.tref": t}
            for r, t in [(0.1, 9.0), (0.5, 10.0), (0.9, 11.0)]
        ]
        modes = [
            {"aerosol": Aerosol(), "params.do_seeding": False},
            {"params.do_seeding": False},
        ]
        reps = [{"params.n_blob": nb} for nb in (1, 2, 5)]

        design = cross(lhs, modes, reps)
        cases = base.sweep(design)

        assert len(design) == 18
        assert len(cases) == 18
        assert len({c.name for c in cases}) == 18


class TestDefaultName:
    def test_pads_to_three_digits(self) -> None:
        assert default_name("sweep", 0, 4) == "sweep_000"
        assert default_name("sweep", 12, 100) == "sweep_012"

    def test_width_follows_the_largest_index(self) -> None:
        assert default_name("sweep", 7, 1000) == "sweep_007"    # 0..999
        assert default_name("sweep", 7, 1001) == "sweep_0007"   # 0..1000

    def test_uses_the_base_name(self) -> None:
        assert default_name("EXP002_sf01", 3, 250).startswith("EXP002_sf01_")


class TestApplyPointFunction:
    """apply_point is public: Run and Registry code may reuse it."""

    def test_matches_the_method(self) -> None:
        a, b = Case(), Case()
        a.apply({"params.tref": 22.0})
        apply_point(b, {"params.tref": 22.0})
        assert a.params.get("tref") == b.params.get("tref")
