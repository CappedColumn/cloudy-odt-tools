"""An optional list of the simulations that were run.

Nothing else in codt_tools imports this. Building a case, staging a run,
launching a batch and analyzing output all work without it — recording a run
is a separate, explicit act.

>>> from codt_tools.registry import Registry
>>> with Registry("~/codt_runs.db") as reg:
...     reg.add_many(runs, tags="EXP005")
"""

from codt_tools.registry.store import COLUMNS, SCHEMA_SQL, Registry

__all__ = [
    "COLUMNS",
    "SCHEMA_SQL",
    "Registry",
]
