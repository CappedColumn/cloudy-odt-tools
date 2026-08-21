"""``Case`` — a complete CODT input description, with no file locations in it.

A case is *what to simulate*: namelist parameters, the aerosol population, and
the parcel trajectory. It is deliberately not *where* to simulate it. The four
namelist keys that point CODT at data are owned by staging, not by the case:

===================  =========================  ==================================
key                  CODT resolves it against   who sets it here
===================  =========================  ==================================
``output_directory`` the **process cwd**        :meth:`Case.write_inputs`
``aerosol_file``     the namelist's directory   :meth:`Case.write_inputs`
``parcel_file``      the namelist's directory   :meth:`Case.write_inputs`
``mie_data_file``    the namelist's directory   user sets the *source*; staged
===================  =========================  ==================================

(CODT: ``app/main.f90:43``, ``src/globals.f90:417``, ``src/initialize.f90:246``.)

So :meth:`Case.set` refuses the first three, the constructors normalize them
away after using them to find files, and :meth:`write_inputs` assigns all four
from its own arguments every time it runs — never from values carried on the
case. Writing a case twice to two directories gives two independent runs.
"""

from __future__ import annotations

import copy
import shutil
from itertools import product
from pathlib import Path
from typing import Any, Union

from codt_tools.case.aerosol import Aerosol
from codt_tools.case.namelist import Namelist
from codt_tools.case.parcel import Parcel
from codt_tools.case.validate import initial_launch_level, validate_case

#: Canonical names the input files are staged under. CODT resolves them
#: against the namelist's own directory, so bare names are correct for any
#: invocation that passes an absolute ``params.nml`` path.
AEROSOL_FILENAME: str = "aerosol_input.nc"
PARCEL_FILENAME: str = "parcel_input.nc"


def _get(nml: Namelist, key: str, default: Any) -> Any:
    """``nml.get`` for keys a file-loaded namelist may simply not carry."""
    try:
        return nml.get(key)
    except KeyError:
        return default


#: Namelist keys that only :meth:`Case.write_inputs` may set.
_STAGED_KEYS: frozenset[str] = frozenset(
    {"output_directory", "aerosol_file", "parcel_file"}
)


class Case:
    """A complete set of CODT inputs: namelist, aerosol, parcel trajectory.

    Attributes
    ----------
    params : Namelist
        Namelist parameters.
    aerosol : Aerosol
        Aerosol population and injection schedule (``aerosol_input.nc``).
    parcel : Parcel
        Waypoint-leg trajectory and optional sounding (``parcel_input.nc``);
        unused in chamber mode.

    Examples
    --------
    >>> case = Case()
    >>> case.set(simulation_name="my_sim", tref=22.0, tmax=7200.0)
    >>> case.aerosol.set(injection_rate=1.0e5)
    >>> case.validate()
    >>> case.write_inputs("/scratch/my_sim/inputs")

    Start from an existing input directory:

    >>> case = Case.from_input_dir("~/dev/CODT/input")
    >>> case.params.get("tref")
    21.5
    """

    #: Attributes that live on the instance itself. Anything else is rejected
    #: rather than silently created, so a mistyped parameter name is an error
    #: instead of a dead attribute.
    _OWN_ATTRS: frozenset[str] = frozenset({"params", "aerosol", "parcel"})

    def __init__(self) -> None:
        self.params: Namelist = Namelist()
        self.aerosol: Aerosol = Aerosol()
        self.parcel: Parcel = Parcel()

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_input_dir(cls, path: Union[str, Path]) -> "Case":
        """Build a case from a directory of CODT input files.

        Reads ``params.nml`` and, from the same directory, whichever files its
        ``aerosol_file`` / ``parcel_file`` name. Those paths are used to *find*
        the data and are then normalized away — the returned case carries the
        contents, not the locations, so writing it cannot aim a new run at the
        directory it was read from.

        Parameters
        ----------
        path : str or Path
            A directory containing ``params.nml``, or the ``params.nml`` path
            itself (its parent directory is used).

        Returns
        -------
        Case

        Raises
        ------
        FileNotFoundError
            If no ``params.nml`` is found.
        """
        path = Path(path).expanduser()
        nml_path = path if path.suffix == ".nml" else path / "params.nml"
        if not nml_path.is_file():
            raise FileNotFoundError(f"Namelist file not found: {nml_path}")
        input_dir = nml_path.resolve().parent

        obj = cls.__new__(cls)
        obj.params = Namelist(nml_path)

        aerosol_path = input_dir / str(obj.params.get("aerosol_file"))
        obj.aerosol = Aerosol(
            aerosol_path if aerosol_path.is_file() else None
        )

        try:
            parcel_file = str(obj.params.get("parcel_file"))
        except KeyError:
            parcel_file = ""
        parcel_path = input_dir / parcel_file if parcel_file else None
        obj.parcel = Parcel(
            parcel_path if parcel_path and parcel_path.is_file() else None
        )

        obj._clear_staged_keys()
        return obj

    @classmethod
    def from_simulation(
        cls,
        path: Union[str, Path],
        run_dir: Union[str, Path, None] = None,
    ) -> "Case":
        """Build a case from a completed simulation's output.

        Recovers the configuration from:

        - **Namelist parameters** from netCDF global attributes (always
          available in CODT v0.4+).
        - **DSD bin edges** from the ``radius_edges`` variable in the
          output netCDF.
        - **Aerosol / parcel input** from the run directory (required — CODT
          v0.5.x no longer copies input files to the output directory).

        The recovered path keys are normalized away, as in
        :meth:`from_input_dir`: a case rebuilt from a finished run cannot
        inherit that run's output directory.

        Parameters
        ----------
        path : str or Path
            Path to the output ``.nc`` file, or a directory containing
            exactly one.
        run_dir : str or Path
            Path to the run directory containing ``aerosol_input.nc`` and
            optionally ``parcel_input.nc``.

        Returns
        -------
        Case

        Raises
        ------
        FileNotFoundError
            If *run_dir* is not provided or does not contain
            ``aerosol_input.nc``.

        Examples
        --------
        >>> case = Case.from_simulation(
        ...     "output/old_run.nc",
        ...     run_dir="output/old_run/inputs",
        ... )
        """
        from codt_tools.simulation import CODTSimulation

        sim = CODTSimulation(path)

        obj = cls.__new__(cls)

        # Namelist: prefer the simulation's parsed params (from netCDF attrs)
        if sim.params is not None:
            obj.params = copy.deepcopy(sim.params)
        else:
            obj.params = Namelist()

        if run_dir is None:
            sim.close()
            raise FileNotFoundError(
                "run_dir is required: CODT v0.5.x does not copy input "
                "files to the output directory. Pass the path to the "
                "run's input directory (e.g. '{base}/{name}/inputs/')."
            )

        run_dir = Path(run_dir)
        aerosol_path = run_dir / AEROSOL_FILENAME
        if not aerosol_path.is_file():
            sim.close()
            raise FileNotFoundError(
                f"{AEROSOL_FILENAME} not found in run_dir: {run_dir}. "
                f"Expected: {aerosol_path}"
            )
        obj.aerosol = Aerosol(aerosol_path)

        parcel_path = run_dir / PARCEL_FILENAME
        obj.parcel = Parcel(parcel_path if parcel_path.is_file() else None)

        # DSD bin edges: prefer radius_edges from the netCDF output
        try:
            edges = sim.bin_edges
            obj.aerosol.set(dsd_bin_edges=edges)
        except (KeyError, AttributeError):
            pass

        sim.close()
        obj._clear_staged_keys()
        return obj

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def set(self, **kwargs: Any) -> None:
        """Set namelist parameters by name, routed to the correct group.

        Delegates to :meth:`Namelist.set`, refusing the keys that name file
        locations — those are set when the case is written.

        Parameters
        ----------
        **kwargs
            Parameter name-value pairs (case-insensitive).

        Raises
        ------
        ValueError
            If a staged path key is given.
        KeyError
            If a parameter name is not a CODT namelist parameter.

        Examples
        --------
        >>> case.set(tref=22.0, tmax=7200.0, simulation_name="test")
        """
        staged = sorted(k for k in kwargs if k.lower() in _STAGED_KEYS)
        if staged:
            raise ValueError(
                f"{', '.join(staged)} name file locations, which are set when "
                f"the case is written, not on the case itself: "
                f"case.write_inputs(directory, output_directory=...). Data "
                f"files are always staged as {AEROSOL_FILENAME} / "
                f"{PARCEL_FILENAME} beside params.nml."
            )
        self.params.set(**kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject attributes that are not part of a case.

        ``case.tmax = 100`` used to create a dead attribute that no run ever
        saw. It now raises, naming :meth:`set`.
        """
        if name in self._OWN_ATTRS:
            object.__setattr__(self, name, value)
            return
        try:
            self.params._find_group(name)
        except (AttributeError, KeyError):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                f"Case holds params, aerosol and parcel."
            ) from None
        raise AttributeError(
            f"'{name}' is a namelist parameter: set it with "
            f"case.set({name}=...)."
        )

    @property
    def name(self) -> str:
        """Simulation name (from namelist ``simulation_name``)."""
        return self.params.get("simulation_name")

    def _clear_staged_keys(self) -> None:
        """Reset the path keys to their neutral, unstaged values."""
        self.params.set(
            output_directory="",
            aerosol_file=AEROSOL_FILENAME,
            parcel_file="",
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def staged_namelist(
        self,
        directory: Union[str, Path],
        output_directory: Union[str, Path, None] = None,
    ) -> Namelist:
        """The namelist as it will be written into *directory*.

        The only place the path keys are assigned. Returns a copy; ``self`` is
        untouched.

        Parameters
        ----------
        directory : str or Path
            Where the input files go. The data files are referenced by bare
            name, which CODT resolves against the namelist's own directory.
        output_directory : str or Path, optional
            Where CODT writes results. Resolved to an absolute path — a
            relative one would be resolved against whatever directory the job
            starts in. Defaults to ``directory/"output"``.
        """
        directory = Path(directory).expanduser().resolve()
        if output_directory is None:
            output_directory = directory / "output"
        output_directory = Path(output_directory).expanduser().resolve()

        nml = copy.deepcopy(self.params)
        nml.set(
            aerosol_file=AEROSOL_FILENAME,
            output_directory=str(output_directory),
        )
        if _get(nml, "simulation_mode", "chamber") == "parcel":
            nml.set(parcel_file=PARCEL_FILENAME)
        else:
            nml.set(parcel_file="")
        mie_source = str(_get(nml, "mie_data_file", "")).strip()
        if _get(nml, "do_radiation", False) and mie_source:
            nml.set(mie_data_file=Path(mie_source).name)
        return nml

    def write_inputs(
        self,
        directory: Union[str, Path],
        output_directory: Union[str, Path, None] = None,
    ) -> Namelist:
        """Write this case's input files into *directory*.

        Writes ``aerosol_input.nc``, ``parcel_input.nc`` (parcel mode), the
        Mie table (when ``do_radiation``), and ``params.nml`` last so the
        namelist's path checks see the files. Nothing on the case is modified.

        Parameters
        ----------
        directory : str or Path
            Target directory for the input files. Created if needed.
        output_directory : str or Path, optional
            Where CODT should write results; created so CODT's
            parent-must-exist check cannot fail. Defaults to
            ``directory/"output"``.

        Returns
        -------
        Namelist
            The namelist that was written — the staged paths included. Callers
            that record provenance should use this rather than ``case.params``.

        Examples
        --------
        >>> staged = case.write_inputs(run_dir / "inputs",
        ...                            output_directory=run_dir / "output")
        >>> case.params.get("output_directory")     # unchanged
        ''
        """
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)

        staged = self.staged_namelist(directory, output_directory)
        Path(staged.get("output_directory")).mkdir(parents=True, exist_ok=True)

        self.aerosol.write(directory / AEROSOL_FILENAME)

        if _get(staged, "simulation_mode", "chamber") == "parcel":
            self.parcel.write(
                directory / PARCEL_FILENAME,
                initial_level=initial_launch_level(staged, self.parcel),
                vertical_axis=staged.get("vertical_axis"),
                n_grid=staged.get("n"),
            )

        if _get(staged, "do_radiation", False):
            mie_source = Path(
                str(self.params.get("mie_data_file")).strip()
            ).expanduser()
            if not mie_source.is_file():
                raise FileNotFoundError(
                    f"mie_data_file not found: {mie_source}. do_radiation "
                    f"needs the Mie table on disk so it can be staged next "
                    f"to params.nml."
                )
            shutil.copyfile(mie_source, directory / mie_source.name)

        staged.write(directory / "params.nml")
        return staged

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check the case against CODT's own startup guards.

        See :func:`codt_tools.case.validate.validate_case`.

        Raises
        ------
        ValueError
            If any consistency check fails.
        """
        validate_case(self)

    # ------------------------------------------------------------------
    # Copy / sweep
    # ------------------------------------------------------------------

    def copy(self) -> "Case":
        """Return a deep copy of this case."""
        return copy.deepcopy(self)

    # Short-name abbreviations for sweep-generated simulation names.
    _SWEEP_ABBREV: dict[str, str] = {
        "tref": "Tref",
        "tdiff": "Tdiff",
        "tmax": "tmax",
        "volume_scaling": "VS",
        "n": "N",
        "h": "H",
        "pres": "P",
        "lmin": "Lmin",
        "lprob": "Lprob",
        "max_accept_prob": "MAP",
        "c2": "C2",
        "zc2": "ZC2",
        "expected_ndrops_per_gridpoint": "Ndrops",
        "initial_wet_radius": "Rw",
        "aerosol_concentration": "Naer",
        "write_timer": "dt",
        "integral_length_scale": "Lint",
        "dissipation_rate": "eps",
        "initial_rh": "RH",
    }

    @staticmethod
    def sweep(base: "Case", **param_ranges: list) -> list["Case"]:
        """Generate a Cartesian product of namelist parameter variations.

        Each returned case is an independent deep copy with a unique
        ``simulation_name`` built from the base name and the varied values.

        Parameters
        ----------
        base : Case
            Base case to vary.
        **param_ranges
            Keyword arguments mapping parameter names to lists of values.

        Returns
        -------
        list[Case]
            One case per combination.

        Examples
        --------
        >>> cases = Case.sweep(base, tref=[20.0, 21.0], volume_scaling=[13, 50])
        >>> len(cases)
        4
        >>> cases[0].name
        'default_sim_Tref20.0_VS13'
        """
        if not param_ranges:
            return [base.copy()]

        param_names = list(param_ranges.keys())
        param_values = [param_ranges[k] for k in param_names]
        base_name = base.name

        cases: list[Case] = []
        for combo in product(*param_values):
            case = base.copy()
            parts: list[str] = []
            for name, value in zip(param_names, combo):
                case.set(**{name: value})
                abbrev = Case._SWEEP_ABBREV.get(name.lower(), name)
                parts.append(f"{abbrev}{value}")
            case.set(simulation_name=f"{base_name}_{'_'.join(parts)}")
            cases.append(case)

        return cases

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Case(name='{self.name}', aerosol={self.aerosol!r})"
