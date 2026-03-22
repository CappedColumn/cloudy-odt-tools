"""Tests for codt_tools.trajectory_io (particle trajectory reader)."""

from __future__ import annotations

import numpy as np
import pytest

from codt_tools.trajectory_io import (
    load_particles,
    particles_at_timestep,
    record_times,
    trajectory_of,
    unique_particle_ids,
)


class TestLoadParticles:

    def test_returns_dataset(self, particles_nc):
        ds = load_particles(particles_nc)
        assert "particle_id" in ds
        assert "row_sizes" in ds
        assert "time" in ds

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_particles(tmp_path / "nonexistent.nc")


class TestParticlesAtTimestep:

    def test_correct_count(self, particles_nc):
        ds = load_particles(particles_nc)
        subset = particles_at_timestep(ds, 0)
        row_sizes = ds["row_sizes"].values
        assert len(subset.record) == row_sizes[0]

    def test_all_timesteps_sum(self, particles_nc):
        ds = load_particles(particles_nc)
        total = sum(
            len(particles_at_timestep(ds, t).record)
            for t in range(len(ds.time_step))
        )
        assert total == len(ds.record)

    def test_out_of_range(self, particles_nc):
        ds = load_particles(particles_nc)
        with pytest.raises(IndexError):
            particles_at_timestep(ds, 999)

    def test_negative_index(self, particles_nc):
        ds = load_particles(particles_nc)
        with pytest.raises(IndexError):
            particles_at_timestep(ds, -1)


class TestTrajectoryOf:

    def test_returns_single_particle(self, particles_nc):
        ds = load_particles(particles_nc)
        pids = unique_particle_ids(ds)
        traj = trajectory_of(ds, int(pids[0]))
        assert np.all(traj["particle_id"].values == pids[0])

    def test_trajectory_length(self, particles_nc):
        """Each particle appears at every timestep in the fixture."""
        ds = load_particles(particles_nc)
        pids = unique_particle_ids(ds)
        traj = trajectory_of(ds, int(pids[0]))
        n_timesteps = len(ds["time_step"])
        assert len(traj.record) == n_timesteps

    def test_nonexistent_pid(self, particles_nc):
        ds = load_particles(particles_nc)
        with pytest.raises(ValueError, match="No records found"):
            trajectory_of(ds, 999999)


class TestUniqueParticleIds:

    def test_sorted(self, particles_nc):
        ds = load_particles(particles_nc)
        pids = unique_particle_ids(ds)
        np.testing.assert_array_equal(pids, np.sort(pids))

    def test_expected_ids(self, particles_nc):
        """Fixture creates particles with IDs 1..n_particles."""
        ds = load_particles(particles_nc)
        pids = unique_particle_ids(ds)
        np.testing.assert_array_equal(pids, [1, 2, 3, 4])


class TestRecordTimes:

    def test_length_matches_records(self, particles_nc):
        ds = load_particles(particles_nc)
        rt = record_times(ds)
        assert len(rt) == len(ds.record)

    def test_values_match_timestep_times(self, particles_nc):
        ds = load_particles(particles_nc)
        rt = record_times(ds)
        row_sizes = ds["row_sizes"].values
        times = ds["time"].values

        # First row_sizes[0] records should all have time = times[0]
        np.testing.assert_allclose(rt[:row_sizes[0]], times[0])
        # Last row_sizes[-1] records should have time = times[-1]
        np.testing.assert_allclose(rt[-row_sizes[-1]:], times[-1])
