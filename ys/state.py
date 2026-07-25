import json
import os
from typing import Optional

from ys import paths


class RunAlreadyActive(Exception):
    pass


def get_active() -> Optional[dict]:
    if not os.path.exists(paths.ACTIVE_RUN_PATH):
        return None
    with open(paths.ACTIVE_RUN_PATH) as f:
        return json.load(f)


def set_active(run_id: str, experiment: str, arm: str, started_at: str, force: bool = False):
    paths.ensure_home()
    existing = get_active()
    if existing and not force:
        raise RunAlreadyActive(
            f"run {existing['run_id']} (exp={existing['experiment']}, arm={existing['arm']}) "
            "is already active. Use --force to override, or `ys end` it first."
        )
    with open(paths.ACTIVE_RUN_PATH, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "experiment": experiment,
                "arm": arm,
                "started_at": started_at,
            },
            f,
            indent=2,
        )


def clear_active():
    if os.path.exists(paths.ACTIVE_RUN_PATH):
        os.remove(paths.ACTIVE_RUN_PATH)
