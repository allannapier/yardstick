"""Filesystem helpers for the experiment YAMLs the dashboard creates/lists.
Convention: one file per experiment, named `<experiment-name>.yaml`, inside
ys.paths.EXPERIMENTS_DIR. CLI-authored experiments living elsewhere on disk
(e.g. this repo's experiments/example.yaml) aren't discovered by the
dashboard's list view -- only ones it (or you) put in EXPERIMENTS_DIR.
"""
import os
import re

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
    validate_name(name)
    return os.path.join(paths.EXPERIMENTS_DIR, f"{name}.yaml")


def list_experiments() -> list[Experiment]:
    paths.ensure_home()
    out = []
    if not os.path.isdir(paths.EXPERIMENTS_DIR):
        return out
    for fname in sorted(os.listdir(paths.EXPERIMENTS_DIR)):
        if not fname.endswith(".yaml"):
            continue
        try:
            out.append(load_experiment(os.path.join(paths.EXPERIMENTS_DIR, fname)))
        except Exception:
            continue  # skip unparsable files rather than 500ing the whole list
    return out


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


def read_raw(name: str) -> str:
    with open(experiment_path(name)) as f:
        return f.read()
