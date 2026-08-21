"""Building ensembles: points, designs, and the paths that address a case.

The vocabulary is deliberately small, because real ensembles are not simple
Cartesian products and a sweep DSL that tried to express them would be harder
to debug than the ensemble itself.

- A **point** is a plain ``dict`` of ``path -> value``: one case's worth of
  changes. ``{"params.tref": 22.0, "aerosol": seeded}``
- A **design** is a list of points: one entry per run.

Only two things need library support — crossing axes (:func:`cross`) and
applying a point across the three components (:func:`apply_point`). Everything
else an ensemble needs is ordinary Python on a list of dicts:

===============================  ==========================================
need                             expressed as
===============================  ==========================================
cross axes                       ``cross(a, b, c)``
values that vary together        one axis *is* a list of dicts
  (LHS samples, paired params)
values derived from others       computed where the axis is built
a control group beside a sweep   ``design_a + design_b``
drop invalid combinations        a list comprehension
swap a whole component           ``{"aerosol": <Aerosol>}``
inspect the design               it is a list of dicts
===============================  ==========================================
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from codt_tools.case.case import Case

#: Components of a case a path may address.
COMPONENTS: tuple[str, ...] = ("params", "aerosol", "parcel")

#: A point: path -> value. A design is a list of these.
Point = dict[str, Any]


# ======================================================================
# Applying a point
# ======================================================================


def apply_point(case: "Case", point: Mapping[str, Any]) -> None:
    """Apply one point to *case*, in place.

    Parameters
    ----------
    case : Case
        The case to modify.
    point : mapping
        ``path -> value``. A path is ``"<component>.<field>"`` to set one
        field, or a bare ``"<component>"`` to replace the whole component.
        A **callable** value is called with the current value and its return
        used, so a point can transform rather than overwrite.

    Raises
    ------
    ValueError
        On an unknown or unprefixed path, on a staged path key (those belong
        to :meth:`Case.write_inputs`), or when a callable returns ``None``.

    Examples
    --------
    >>> case.apply({"params.tref": 22.0, "parcel.velocity": [1.0]})
    >>> case.apply({"params.tmax": lambda t: 2 * t})
    >>> case.apply({"aerosol": prebuilt_aerosol})
    """
    for path, value in point.items():
        component, _, field = _split(path)

        if field is None:
            current = getattr(case, component)
            new = _resolve(value, current, path)
            _check_component_type(case, component, new, path)
            setattr(case, component, new)
            continue

        target = getattr(case, component)
        if component == "params":
            # Route through Case.set so the staged-key guard applies.
            current = _get_param(case, field)
            case.set(**{field: _resolve(value, current, path)})
        else:
            current = getattr(target, field, None)
            target.set(**{field: _resolve(value, current, path)})


def _split(path: str) -> tuple[str, str, str | None]:
    """Split ``"aerosol.injection_rate"`` into its component and field."""
    if not isinstance(path, str):
        raise ValueError(
            f"Sweep paths must be strings, got {type(path).__name__}: {path!r}"
        )

    component, sep, field = path.partition(".")
    if component not in COMPONENTS:
        if not sep:
            raise ValueError(
                f"'{path}' has no component prefix. Say which part of the "
                f"case it addresses: one of "
                f"{', '.join(f'{c}.{path}' for c in COMPONENTS)}."
            )
        raise ValueError(
            f"'{path}' does not address a case: '{component}' is not one of "
            f"{', '.join(COMPONENTS)}."
        )
    return component, sep, (field if sep else None)


def _resolve(value: Any, current: Any, path: str) -> Any:
    """A callable value transforms the current value; anything else replaces it."""
    if not callable(value):
        return value
    new = value(current)
    if new is None:
        raise ValueError(
            f"The callable for '{path}' returned None. Return the new value "
            f"— modifying in place and returning nothing would blank it."
        )
    return new


def _get_param(case: "Case", field: str) -> Any:
    """A namelist parameter's current value, or None if it has none yet."""
    try:
        return case.params.get(field)
    except KeyError:
        return None


def _check_component_type(
    case: "Case", component: str, value: Any, path: str
) -> None:
    expected = type(getattr(case, component))
    if not isinstance(value, expected):
        raise ValueError(
            f"'{path}' replaces the whole {component}, so it needs "
            f"a {expected.__name__}, got {type(value).__name__}."
        )


# ======================================================================
# Building a design
# ======================================================================


def cross(*axes: Iterable[Mapping[str, Any]] | Mapping[str, Sequence[Any]]) -> list[Point]:
    """Cartesian product of axes, merged point-wise.

    Each axis is either a list of points (whose values therefore vary
    *together* — how an LHS sample or a paired ``(rep, n_blob)`` ladder is
    expressed), or a ``dict`` of ``path -> list of values``, which expands into
    one independent axis per key.

    Parameters
    ----------
    *axes
        The axes to cross. No axes gives ``[{}]`` — the empty product, one
        point that changes nothing.

    Returns
    -------
    list[dict]
        One point per combination.

    Raises
    ------
    ValueError
        If two crossed axes set the same path. Crossed axes are meant to be
        independent, so that is a design bug rather than an override; use
        ``design_a + design_b`` to union two designs.

    Examples
    --------
    >>> cross({"params.tref": [20.0, 22.0]}, {"params.n_blob": [1, 5]})
    [{'params.tref': 20.0, 'params.n_blob': 1}, ...]

    An LHS sample crossed with a mode axis — 10 x 5, not 10**4 x 5:

    >>> design = cross(lhs_points, mode_points)
    """
    expanded: list[list[Point]] = []
    for axis in axes:
        expanded.extend(_as_axes(axis))

    design: list[Point] = []
    for combo in product(*expanded):
        merged: Point = {}
        for part in combo:
            clash = sorted(set(part) & set(merged))
            if clash:
                raise ValueError(
                    f"Crossed axes both set {', '.join(clash)}. Crossed axes "
                    f"must be independent; to put two designs side by side, "
                    f"add them: design_a + design_b."
                )
            merged.update(part)
        design.append(merged)
    return design


def _as_axes(axis: Any) -> list[list[Point]]:
    """Normalize one argument of :func:`cross` into a list of axes."""
    if isinstance(axis, Mapping):
        # dict of path -> values: one independent axis per key.
        return [
            [{path: value} for value in _as_sequence(path, values)]
            for path, values in axis.items()
        ]

    points = list(axis)
    for point in points:
        if not isinstance(point, Mapping):
            raise ValueError(
                f"An axis is a list of points (path -> value dicts) or a dict "
                f"of path -> values, got a {type(point).__name__} in the list."
            )
    return [points]


def _as_sequence(path: str, values: Any) -> Sequence[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(
            f"'{path}' needs a list of values to sweep over, got "
            f"{type(values).__name__}. A single value belongs on the base case."
        )
    return list(values)


def as_design(design: Any) -> list[Point]:
    """Normalize a design: a list of points, or a dict-of-lists to cross."""
    if isinstance(design, Mapping):
        return cross(design)

    points = list(design)
    for point in points:
        if not isinstance(point, Mapping):
            raise ValueError(
                f"A design is a list of points (path -> value dicts), got a "
                f"{type(point).__name__}."
            )
    return points


# ======================================================================
# Naming
# ======================================================================


def default_name(base_name: str, index: int, total: int) -> str:
    """``{base_name}_{index}``, zero-padded to at least three digits.

    Sweep names become run *directory* names, so the default is an index: it
    is always path-safe and never collides, however many runs there are and
    whatever they vary. What each index means is recorded by the run's own
    staged input files and by the registry — deliberately not here.

    Pass ``name=`` to :meth:`Case.sweep` for a project's own convention.
    """
    width = max(3, len(str(max(total - 1, 0))))
    return f"{base_name}_{index:0{width}d}"


def resolve_names(
    base_name: str,
    design: Sequence[Mapping[str, Any]],
    name: Callable[[int, Mapping[str, Any]], str] | None,
) -> list[str]:
    """One name per point, checked for duplicates."""
    total = len(design)
    if name is None:
        return [default_name(base_name, i, total) for i in range(total)]

    names = [str(name(i, dict(point))) for i, point in enumerate(design)]
    seen: dict[str, int] = {}
    for i, value in enumerate(names):
        if value in seen:
            raise ValueError(
                f"name= produced '{value}' for both point {seen[value]} and "
                f"point {i}. Run names become directory names, so they have "
                f"to be unique."
            )
        seen[value] = i
    return names
