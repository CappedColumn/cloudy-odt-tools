"""Tests for codt_tools.plotting utilities."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import xarray as xr
import numpy as np
import pytest

from codt_tools.plotting import comparison_colors, _get_label


class TestComparisonColors:

    def test_returns_correct_count(self):
        assert len(comparison_colors(1)) == 1
        assert len(comparison_colors(3)) == 3
        assert len(comparison_colors(7)) == 7

    def test_returns_valid_hex(self):
        colors = comparison_colors(4)
        for c in colors:
            assert c.startswith("#")
            # Should be parseable by matplotlib
            mcolors.to_rgba(c)

    def test_colors_are_distinct(self):
        colors = comparison_colors(3)
        assert len(set(colors)) == 3

    def test_single_color_is_midpoint(self):
        colors = comparison_colors(1)
        assert len(colors) == 1

    def test_custom_cmap(self):
        colors_v = comparison_colors(3, cmap="viridis")
        colors_p = comparison_colors(3, cmap="plasma")
        assert colors_v != colors_p

    def test_large_n(self):
        colors = comparison_colors(20)
        assert len(colors) == 20


class TestGetLabel:

    def test_long_name_underscore(self):
        da = xr.DataArray(
            [1, 2, 3],
            attrs={"long_name": "Temperature", "units": "celsius"},
        )
        assert _get_label(da) == "Temperature (celsius)"

    def test_long_name_space_fallback(self):
        da = xr.DataArray(
            [1, 2, 3],
            attrs={"long name": "Temperature", "units": "celsius"},
        )
        assert _get_label(da) == "Temperature (celsius)"

    def test_underscore_takes_priority(self):
        da = xr.DataArray(
            [1, 2, 3],
            attrs={
                "long_name": "Correct",
                "long name": "Wrong",
                "units": "K",
            },
        )
        assert _get_label(da) == "Correct (K)"

    def test_no_units(self):
        da = xr.DataArray(
            [1, 2, 3],
            attrs={"long_name": "Count"},
        )
        assert _get_label(da) == "Count"

    def test_fallback_to_name(self):
        da = xr.DataArray([1, 2, 3], name="my_var")
        assert _get_label(da) == "my_var"

    def test_no_attrs_no_name(self):
        da = xr.DataArray([1, 2, 3])
        assert _get_label(da) == ""
