"""`ys export --csv` / `--json` -- turn recorded runs into files that can
leave the tool (feature 6, "smaller additions", in IMPROVEMENTS.md).

Unit of export: **one row per run**, not one row per arm-aggregate and not
one row per request. Reasoning:

- Per-arm aggregates are already `ys compare`/`ys report --html`'s job, and
  `aggregate_run_metrics` (findings 13/14) deliberately *excludes*
  unfinished/abandoned runs and runs from a stale config version before
  computing a mean -- exactly the information an export needs to keep
  visible, not fold away. A per-arm-only export could never answer "was
  this run finished, and does it match today's config?" because that
  detail dies in the aggregation.
- Per-request rows are the finest granularity available, but they don't
  carry "finished/abandoned" or "config group" as a row-level fact (that's
  a property of the run, not the request) and would need a join back to
  `runs` to be interpretable at all -- defeating the point of a
  self-contained export file. They also multiply the row count by every
  request in every run, which is a lot of rows for very little extra
  signal beyond what's already summarized per run.
- A run is the natural unit that already carries its own identifying
  context on the row itself: experiment, arm, run id, `config_hash` (which
  version of the config produced it, finding 14), and status (success /
  fail / unfinished / abandoned, finding 13) -- so a reader doesn't need
  the database beside the file to know what population it's looking at.

This module is deliberately just serialization -- plain dicts in, CSV/JSON
text out -- with no Rich/HTML in it, mirroring the layering `ys/render.py`
already uses for presentation vs `ys/statistics.py` for the actual numbers.
"""
import csv
import io
import json
from typing import Optional

from ys import metrics
from ys.experiment import Experiment
from ys.runs import arm_row_id, config_hash_for_arm

# Stable column order for both CSV (which needs a fixed header) and JSON
# (kept in the same order purely so the two formats are easy to eyeball
# side by side, not because JSON needs one).
COLUMNS = [
    "experiment",
    "arm",
    "run_id",
    "repeat_idx",
    "status",
    "config_hash",
    "config_matches_current",
    "started_at",
    "ended_at",
    "wall_clock_s",
    "manual_score",
    "model",
    "turns",
    "billable_tokens",
    "cost_usd",
    "cost_unknown",
    "tool_calls",
    "tool_error_rate",
    "redundant_tool_calls",
    "cache_read_ratio",
    "context_high_water",
    "compaction_events",
    "background_requests",
    "background_tokens",
    "active_s",
]


def _status(run_row: dict) -> str:
    """Finding 13: `task_success IS NULL` means "never scored" (still
    active, or displaced by `--force` and flagged `abandoned`) -- not the
    same thing as "failed". A reader who only sees a bare `task_success`
    column would read NULL as ambiguous at best; the exported `status`
    column spells out which of the four states this row is in."""
    if run_row.get("abandoned"):
        return "abandoned"
    ts = run_row["task_success"]
    if ts is None:
        return "unfinished"
    return "success" if ts else "fail"


def export_rows(cur, experiment: Experiment, arm_id: Optional[str] = None) -> list[dict]:
    """One row per recorded run of `experiment` -- every arm (or just
    `arm_id`, if given) and every run, including unfinished/abandoned ones
    and ones recorded under an older config hash. `ys compare` narrows to
    the gate-passing, current-config population on purpose (findings
    13/14); an export is a different job -- handing the reader everything
    with enough context (`status`, `config_matches_current`) to narrow it
    themselves, honestly, rather than the file silently doing it for them.
    """
    billable_weights_by_model = {
        key: weights.model_dump() for key, weights in experiment.billable_weights.items()
    }
    arms = [a for a in experiment.arms if arm_id is None or a.id == arm_id]
    rows = []
    for arm in arms:
        row_id = arm_row_id(experiment.experiment, arm.id)
        current_hash = config_hash_for_arm(experiment, arm)
        run_rows = cur.execute(
            "SELECT * FROM runs WHERE arm_id = ? ORDER BY repeat_idx", (row_id,)
        ).fetchall()
        for r in run_rows:
            run = dict(r)
            m = metrics.compute_run_metrics(cur, run["id"], billable_weights_by_model)
            unpriced = metrics.unpriced_models(cur, run["id"])
            rows.append(
                {
                    "experiment": experiment.experiment,
                    "arm": arm.id,
                    "run_id": run["id"],
                    "repeat_idx": run["repeat_idx"],
                    "status": _status(run),
                    "config_hash": run.get("config_hash"),
                    "config_matches_current": run.get("config_hash") == current_hash,
                    "started_at": run.get("started_at"),
                    "ended_at": run.get("ended_at"),
                    "wall_clock_s": run.get("wall_clock_s"),
                    "manual_score": run.get("manual_score"),
                    "model": run.get("model"),
                    "turns": m.get("turns"),
                    "billable_tokens": m.get("billable_tokens"),
                    "cost_usd": m.get("cost_usd"),
                    "cost_unknown": bool(unpriced),
                    "tool_calls": m.get("tool_calls"),
                    "tool_error_rate": m.get("tool_error_rate"),
                    "redundant_tool_calls": m.get("redundant_tool_calls"),
                    "cache_read_ratio": m.get("cache_read_ratio"),
                    "context_high_water": m.get("context_high_water"),
                    "compaction_events": m.get("compaction_events"),
                    "background_requests": m.get("background_requests"),
                    "background_tokens": m.get("background_tokens"),
                    "active_s": m.get("active_s"),
                }
            )
    return rows


def to_csv(rows: list[dict]) -> str:
    """`csv.DictWriter`'s default `QUOTE_MINIMAL` already quotes any field
    containing the delimiter, a quote character, or a newline -- nothing
    bespoke needed for a model id or experiment name with a comma in it.
    `None` is written as an empty cell (the csv module's own behaviour for
    a `None` field), not the literal string `"None"`."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def to_json(rows: list[dict]) -> str:
    """`default=str` covers nothing today (every value here is already a
    JSON-native type) but keeps this from hard-crashing if a future column
    adds something JSON can't serialize natively -- the same defensive
    choice `ys.db.dumps` already makes."""
    return json.dumps(rows, indent=2, default=str)
