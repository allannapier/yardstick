"""Filesystem helpers for the experiment YAMLs the dashboard reads/writes.

Two kinds of location, deliberately kept distinct:

- `paths.EXPERIMENTS_DIR` (`~/.yardstick/experiments` by default) is the
  dashboard's own writable directory -- the only place `save_experiment`
  ever writes, and the only place `delete`/`edit` (ys/web/app.py) will
  mutate. One file per experiment, named `<experiment-name>.yaml`.
- Everything else `discovery_dirs()` finds (currently: an `experiments/`
  directory next to the process's current working directory, e.g. this
  repo's own `experiments/example.yaml`) is read-only as far as the
  dashboard is concerned. `list_experiments`/`find_experiment` search both,
  so an experiment the CLI already knows about (the file the README's quick
  start points at) shows up without being copied into EXPERIMENTS_DIR
  first -- but the dashboard will not silently rewrite or delete a file
  that's part of a git checkout rather than the user's own scratch config.
  Filenames in a discovered directory need not match the experiment id
  inside (example.yaml's `experiment:` field is `mock-smoke-01`, not
  `example`), so discovery indexes by parsed `experiment.experiment`, not
  by filename.
"""
import os
import re
from typing import Optional

import yaml

from ys import paths
from ys.experiment import Experiment, load_experiment

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class InvalidExperimentName(Exception):
    pass


def validate_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise InvalidExperimentName(
            f"'{name}' is not a valid experiment name -- use letters, digits, "
            "'-' and '_' only, starting with a letter or digit."
        )
    return name


def experiment_path(name: str) -> str:
    """The dashboard's own writable location for `name`. `save_experiment`
    only ever writes here; `find_experiment`/`list_experiments` below widen
    *reading* to other directories, but this function -- and the guarantee
    that it never leaves EXPERIMENTS_DIR -- is unchanged, since every
    caller that mutates a file (save/edit/delete in ys/web/app.py) still
    goes through it. `validate_name` runs first regardless of how many
    directories discovery searches, so a URL-supplied name can never
    resolve outside `<dir>/<validated-name>.yaml` in any of them.
    """
    validate_name(name)
    return os.path.join(paths.EXPERIMENTS_DIR, f"{name}.yaml")


def discovery_dirs() -> list[str]:
    """Directories `list_experiments`/`find_experiment` read from, in
    priority order (earlier wins a name collision). `EXPERIMENTS_DIR` is
    always first, since it's the only directory the dashboard writes to.
    An `experiments/` directory next to the current working directory is
    also searched, read-only -- `ys web up` is ordinarily run from the
    project root, the same convention the CLI's own
    `--exp experiments/example.yaml` relies on, so this is what makes
    the README's quick-start experiment visible in the dashboard without
    an extra copy/symlink step.
    """
    dirs = [paths.EXPERIMENTS_DIR]
    cwd_dir = os.path.abspath("experiments")
    if os.path.isdir(cwd_dir) and os.path.realpath(cwd_dir) != os.path.realpath(
        paths.EXPERIMENTS_DIR
    ):
        dirs.append(cwd_dir)
    return dirs


def _scan(dirpath: str) -> dict:
    """experiment-id -> file path for every parseable *.yaml directly in
    dirpath. Indexed by the parsed `experiment:` field, not the filename,
    since a discovered directory (unlike EXPERIMENTS_DIR) has no guarantee
    the two match."""
    out = {}
    if not os.path.isdir(dirpath):
        return out
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith(".yaml"):
            continue
        full = os.path.join(dirpath, fname)
        try:
            out[load_experiment(full).experiment] = full
        except Exception:
            continue  # skip unparsable files rather than 500ing the whole list
    return out


def _index() -> dict:
    """name -> path across every discovery_dirs() entry. Scanned in
    reverse priority order and overwritten forwards, so a name that exists
    in more than one directory resolves to the highest-priority one
    (EXPERIMENTS_DIR first)."""
    index: dict = {}
    for d in reversed(discovery_dirs()):
        index.update(_scan(d))
    return index


def list_experiments() -> list[Experiment]:
    paths.ensure_home()
    out = []
    for _name, path in sorted(_index().items()):
        try:
            out.append(load_experiment(path))
        except Exception:
            continue
    return out


def find_experiment(name: str) -> Optional[tuple]:
    """Resolve `name` to `(path, Experiment)`, searching discovery_dirs()
    in priority order. Returns None if no directory has it. `name` is
    validated first -- letters/digits/-/_ only -- so widening how many
    directories are searched never widens what a URL can make this open:
    every candidate is still `<one of a fixed, server-side directory
    list>/<validated-name>.yaml`.
    """
    validate_name(name)
    path = _index().get(name)
    if path is None:
        return None
    try:
        return path, load_experiment(path)
    except Exception:
        return None


def is_managed(path: str) -> bool:
    """True if `path` is inside EXPERIMENTS_DIR -- the dashboard's own
    writable directory. Edit and delete (ys/web/app.py) refuse to act on
    anything else, so the dashboard can never rewrite or remove a file
    that lives in, say, a git checkout's experiments/ directory just
    because discovery made it visible."""
    exp_dir = os.path.realpath(paths.EXPERIMENTS_DIR)
    real = os.path.realpath(path)
    return os.path.commonpath([real, exp_dir]) == exp_dir


def save_experiment(data: dict) -> Experiment:
    """Validate `data` as an Experiment (reuses all of ys.experiment's
    validation -- duplicate arm ids, multiple baselines, etc.) and write it
    to EXPERIMENTS_DIR/<name>.yaml. Raises pydantic.ValidationError or
    InvalidExperimentName on bad input; never partially writes a file."""
    experiment = Experiment.model_validate(data)
    path = experiment_path(experiment.experiment)
    paths.ensure_home()
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return experiment


def delete_experiment_file(name: str) -> str:
    """Remove `name`'s definition file. Only ever called after the route
    has confirmed `is_managed(path)` -- see ys/web/app.py's delete_experiment
    -- so this never touches a discovered, non-EXPERIMENTS_DIR file."""
    path = experiment_path(name)
    os.remove(path)
    return path


def read_raw(name: str) -> str:
    """Raw YAML text for `name`, searching the same discovery_dirs() as
    list_experiments/find_experiment -- this is the hook the YAML-view
    route (ys/web/app.py) uses, and it works for a discovered, read-only
    experiment too, not only ones EXPERIMENTS_DIR already has."""
    found = find_experiment(name)
    if found is None:
        raise FileNotFoundError(f"no experiment file found for '{name}'")
    path, _experiment = found
    with open(path) as f:
        return f.read()
