"""Yardstick dashboard: experiment authoring, run start/stop, results
browsing. A thin presentation layer over ys/db.py, ys/runs.py, ys/proxy.py,
and ys/render.py -- it contains no business logic of its own beyond request
parsing and HTML rendering, so it can never drift from what `ys` the CLI
does (see ys/runs.py's docstring for why that split exists).
"""
import json
import os
import sqlite3
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from ys import db, metrics, proxy, render, runs, state
from ys.web import store

app = FastAPI(title="yardstick")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
# Share `ys compare`/`ys report`'s metric formatting rather than re-deriving it
# in Jinja: the templates used a bare "%.4g", which printed six-figure token
# counts as "1.024e+05". See ys/render.py's format_metric.
templates.env.filters["metric"] = render.format_metric


def _write_failed_message(e: Exception) -> str:
    """`runs.begin_run`/`finish_run`/`delete_run` write through
    `db.call_with_retry` (finding 28), which already retries a locked
    database `db.MAX_WRITE_ATTEMPTS` times -- if it still raises, surface a
    plain message on the redirect instead of a 500 traceback."""
    return (
        f"could not write to the database after {db.MAX_WRITE_ATTEMPTS} attempts "
        f"({e}) -- is another yardstick process holding a long write against the "
        "same database file?"
    )


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"


def _redirect(path: str, ok: str = None, error: str = None) -> RedirectResponse:
    params = []
    if ok:
        params.append(f"ok={quote(ok)}")
    if error:
        params.append(f"error={quote(error)}")
    url = path + ("?" + "&".join(params) if params else "")
    return RedirectResponse(url, status_code=303)


def _ctx(request: Request, **extra) -> dict:
    alive, pid = proxy.proxy_status()
    return {
        "request": request,
        "active_run": state.get_active(),
        "proxy_alive": alive,
        "proxy_pid": pid,
        "proxy_port": proxy.read_port(),
        "ok": request.query_params.get("ok"),
        "error": request.query_params.get("error"),
        **extra,
    }


def _error_page(request: Request, status_code: int, message: str) -> HTMLResponse:
    """A human-readable error page with the app's own chrome, instead of a
    raw traceback (defect 19: an invalid experiment name in the URL raised
    `store.InvalidExperimentName` straight through to a 500) or a 303
    redirect to a page that then fails again on its own (defect 23: the
    proxy-up route had no existence check and forwarded to a detail page
    that 500s on the same unguarded `store.experiment_path` call).
    """
    return templates.TemplateResponse(
        request, "error.html", _ctx(request, message=message), status_code=status_code
    )


def _load_experiment_or_404(request: Request, name: str):
    """Resolve `name` to (path, experiment) for every route keyed by
    experiment name, or return a ready-to-serve 404 response. Handles both
    halves of defect 19/23 at the one call site they share: an invalid name
    (`store.validate_name` raising `InvalidExperimentName`) and a
    well-formed but nonexistent one. `store.find_experiment` searches every
    directory `store.discovery_dirs()` knows about (see its docstring) --
    not only EXPERIMENTS_DIR -- so an experiment the CLI already knows
    about (e.g. this repo's own experiments/example.yaml) resolves here
    too, not just ones the dashboard itself created. Returns
    `(path, experiment, None)` on success or `(None, None, error_response)`
    on failure -- callers do `path, experiment, err = ...; if err: return err`.
    """
    try:
        found = store.find_experiment(name)
    except store.InvalidExperimentName as e:
        return None, None, _error_page(request, 404, str(e))
    if found is None:
        return None, None, _error_page(request, 404, f"no experiment named '{name}'")
    path, experiment = found
    return path, experiment, None


def _parse_int_field(form, name: str, default: int, field_errors: dict, label: str) -> int:
    """Guard against defect 20: `int(form.get(...))` on a non-numeric
    timeout_s/repeats raised straight through to a 500. A bad value becomes
    a field-level error instead, and `default` is returned as a placeholder
    -- the caller must check `field_errors` before saving, since this
    function's job is only to keep the rest of the form-processing code
    from crashing on the way to re-rendering the form.
    """
    raw = (form.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        field_errors[name] = f"{label} must be a whole number (got '{raw}')"
        return default


# Pydantic error locations that map onto one specific input in
# new_experiment.html -- everything else (duplicate arm ids, more than one
# baseline, an empty arms list) doesn't correspond to a single field and is
# shown as a general message above the form instead.
_TOP_LEVEL_FIELDS = {
    ("experiment",): "name",
    ("task", "id"): "task_id",
    ("task", "success_check"): "success_check",
    ("task", "timeout_s"): "timeout_s",
    ("repeats",): "repeats",
    ("arms",): "arms",
}


def _split_validation_errors(exc: ValidationError) -> tuple[dict[str, str], list[str]]:
    """Translate pydantic's error list into what defect 22 asked for: short,
    readable messages next to the field they're about, instead of raw
    exception text (`str(exc)`, a multi-line dump including "For further
    information visit https://errors.pydantic.dev/...") URL-encoded into a
    query string.
    """
    field_errors: dict[str, str] = {}
    general: list[str] = []
    for err in exc.errors():
        loc = tuple(err["loc"])
        msg = err["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        field = _TOP_LEVEL_FIELDS.get(loc)
        if field:
            field_errors[field] = msg
        else:
            label = ".".join(str(p) for p in loc)
            general.append(f"{label}: {msg}" if label else msg)
    return field_errors, general


def _extra_factors_by_seq(form) -> dict:
    """Parse the arm-scoped extra-factor rows the arms fieldset emits
    (`factor_arm_seq`/`factor_key`/`factor_value`, one parallel-array
    triple per factor row -- the same shape as `arm_id`/`arm_model`/
    `arm_seq`) into `seq -> [(key, value), ...]`. `arm_seq` is the join key
    between an arm row and its own extra-factor rows, mirroring how
    `arm_baseline` already joins back to a row by `arm_seq` (defect 24) --
    a stable id assigned once per row at creation, not the arm id text
    itself, which the user can still edit freely.
    """
    seqs = form.getlist("factor_arm_seq")
    keys = form.getlist("factor_key")
    values = form.getlist("factor_value")
    out: dict = {}
    for i, seq in enumerate(seqs):
        key = keys[i] if i < len(keys) else ""
        value = values[i] if i < len(values) else ""
        out.setdefault(seq, []).append((key, value))
    return out


def _form_snapshot(form) -> dict:
    """Reconstruct the submitted form shape so new_experiment.html can
    re-populate every field after a validation failure (defects 20/22) --
    the old behaviour redirected to `/experiments/new` and discarded
    everything the user had typed.
    """
    arm_ids = form.getlist("arm_id")
    arm_models = form.getlist("arm_model")
    arm_seqs = form.getlist("arm_seq")
    baseline_seq = form.get("arm_baseline")
    arm_notes = form.getlist("arm_notes")
    extra_by_seq = _extra_factors_by_seq(form)
    arms = []
    for i, arm_id in enumerate(arm_ids):
        seq = arm_seqs[i] if i < len(arm_seqs) else str(i)
        arms.append(
            {
                "id": arm_id,
                "model": arm_models[i] if i < len(arm_models) else "",
                "seq": seq,
                "baseline": baseline_seq is not None and seq == baseline_seq,
                "notes": arm_notes[i] if i < len(arm_notes) else "",
                "extra_factors": [
                    {"key": k, "value": v} for k, v in extra_by_seq.get(seq, [])
                ],
            }
        )

    model_keys = form.getlist("model_key")
    model_kinds = form.getlist("model_kind")
    model_values = form.getlist("model_value")
    models = []
    for i, key in enumerate(model_keys):
        models.append(
            {
                "key": key,
                "kind": model_kinds[i] if i < len(model_kinds) else "mock",
                "value": model_values[i] if i < len(model_values) else "",
            }
        )

    return {
        "name": form.get("name", ""),
        "question": form.get("question", ""),
        "task_id": form.get("task_id", ""),
        "success_check": form.get("success_check", ""),
        "timeout_s": form.get("timeout_s", ""),
        "repeats": form.get("repeats", ""),
        "confirm_overwrite": form.get("confirm_overwrite") == "on",
        "arms": arms,
        "models": models,
    }


def _parse_arms_and_models(form) -> tuple[list, dict]:
    """Shared by create_experiment and edit_experiment -- identical arm/
    model parsing either way; only what happens with the resulting `name`
    differs (refuse-unless-confirmed vs. always-overwrite the same file).

    arm_seq/arm_baseline are a matched pair from the radio group in
    new_experiment.html (defect 24): each row gets a stable seq at
    creation, and the single `arm_baseline` value the browser submits is
    whichever row's radio was checked. Because there's exactly one
    `arm_baseline` field name shared by every row, the browser's native
    radio-group behaviour makes "two arms marked baseline" unreachable
    from the form -- unlike the old checkboxes, which could both be
    checked and only failed later as a raw pydantic error (defect 22).
    """
    arm_ids = form.getlist("arm_id")
    arm_models = form.getlist("arm_model")
    arm_seqs = form.getlist("arm_seq")
    baseline_seq = form.get("arm_baseline")
    arm_notes = form.getlist("arm_notes")
    extra_by_seq = _extra_factors_by_seq(form)

    arms = []
    for i, arm_id in enumerate(arm_ids):
        if not arm_id.strip():
            continue
        seq = arm_seqs[i] if i < len(arm_seqs) else None
        factors = {"model": arm_models[i].strip()} if i < len(arm_models) else {}
        # `Arm.factors` (ys/experiment.py) is an arbitrary dict, but until
        # now only `model` was ever expressible from this form -- the
        # harness-vs-harness comparisons the tool is named for had no UI
        # path at all. Extra key/value rows are scoped to this arm's own
        # `seq`; an empty key is dropped (an unfinished row from the JS
        # "+ add factor" button), and a key of "model" is dropped rather
        # than silently overriding the dedicated model field above.
        for key, value in extra_by_seq.get(seq or "", []):
            key = key.strip()
            if key and key != "model":
                factors[key] = value.strip()
        arms.append(
            {
                "id": arm_id.strip(),
                "factors": factors,
                "baseline": seq is not None and seq == baseline_seq,
                "notes": (arm_notes[i].strip() or None) if i < len(arm_notes) else None,
            }
        )

    model_keys = form.getlist("model_key")
    model_kinds = form.getlist("model_kind")
    model_values = form.getlist("model_value")

    models = {}
    for i, key in enumerate(model_keys):
        if not key.strip():
            continue
        kind = model_kinds[i] if i < len(model_kinds) else "mock"
        value = model_values[i].strip() if i < len(model_values) else ""
        if kind == "mock":
            # The underlying `model` id here never reaches Anthropic --
            # `mock_response` short-circuits it -- but it was still stale
            # (a 2024-era Sonnet 3.5 id). Anything current works equally
            # well for a mock; this just stops the form from teaching a
            # dead model id by example.
            models[key.strip()] = {
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "mock_response": value or "mock response",
            }
        else:
            models[key.strip()] = {
                "model": value if value.startswith("anthropic/") else f"anthropic/{value}",
                "api_key": "os.environ/ANTHROPIC_API_KEY",
            }

    return arms, models


def _experiment_to_form_snapshot(experiment) -> dict:
    """The reverse of `_form_snapshot`: rebuild the form's shape from an
    already-parsed Experiment, so /experiments/{name}/edit can pre-fill
    new_experiment.html with the experiment's current definition instead
    of a bespoke edit UI. `model_kind` is inferred from whether the
    model's litellm_params declare a `mock_response`."""
    arms = []
    for i, arm in enumerate(experiment.arms):
        extra_factors = [
            {"key": k, "value": v} for k, v in arm.factors.items() if k != "model"
        ]
        arms.append(
            {
                "id": arm.id,
                "model": arm.factors.get("model", ""),
                "seq": str(i),
                "baseline": arm.baseline,
                "notes": arm.notes or "",
                "extra_factors": extra_factors,
            }
        )

    models = []
    for key, cfg in experiment.models.items():
        if "mock_response" in cfg:
            models.append({"key": key, "kind": "mock", "value": cfg.get("mock_response", "")})
        else:
            models.append({"key": key, "kind": "real", "value": cfg.get("model", "")})

    return {
        "name": experiment.experiment,
        "question": experiment.question or "",
        "task_id": experiment.task.id,
        "success_check": experiment.task.success_check,
        "timeout_s": experiment.task.timeout_s,
        "repeats": experiment.repeats,
        "confirm_overwrite": False,
        "arms": arms,
        "models": models,
    }


def _new_experiment_response(
    request: Request,
    form_data: dict,
    field_errors: dict,
    general_errors: list,
    status_code: int = 400,
    edit_name: str = None,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "new_experiment.html",
        _ctx(
            request,
            form_data=form_data,
            field_errors=field_errors,
            general_errors=general_errors,
            edit_name=edit_name,
        ),
        status_code=status_code,
    )


# ---------------------------------------------------------------------------
# Experiment list / create
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    experiments = store.list_experiments()
    summaries = []
    with db.cursor() as cur:
        for exp in experiments:
            row = cur.execute(
                "SELECT COUNT(*) AS c FROM runs WHERE experiment_id = ?", (exp.experiment,)
            ).fetchone()
            # store.discovery_dirs() also finds experiments defined outside
            # EXPERIMENTS_DIR (e.g. this repo's own experiments/*.yaml) --
            # `managed` flags whether *this* one is one the dashboard can
            # edit/delete, vs. one it can only view/run against.
            found = store.find_experiment(exp.experiment)
            managed = found is not None and store.is_managed(found[0])
            summaries.append({"experiment": exp, "run_count": row["c"], "managed": managed})
    return templates.TemplateResponse(
        request, "index.html", _ctx(request, summaries=summaries)
    )


@app.get("/experiments/new", response_class=HTMLResponse)
def new_experiment_form(request: Request):
    return templates.TemplateResponse(request, "new_experiment.html", _ctx(request))


@app.post("/experiments")
async def create_experiment(request: Request):
    form = await request.form()
    snapshot = _form_snapshot(form)
    field_errors: dict[str, str] = {}

    timeout_s = _parse_int_field(form, "timeout_s", 1800, field_errors, "timeout (seconds)")
    repeats = _parse_int_field(form, "repeats", 3, field_errors, "repeats per arm")

    name = form.get("name", "").strip()
    path = None
    try:
        store.validate_name(name)
    except store.InvalidExperimentName as e:
        field_errors["name"] = str(e)
    else:
        path = store.experiment_path(name)

    # Defect 21: `store.save_experiment` writes `<name>.yaml` unconditionally,
    # so re-using an existing experiment's name silently overwrote its
    # definition -- the runs already recorded against that experiment id
    # stay attached and get aggregated with whatever the new definition
    # produces. Refuse by default; `confirm_overwrite` (a checkbox the user
    # must tick, never on by default) is the only way past this, so the
    # clobber can't happen by accident.
    if path and os.path.exists(path) and not form.get("confirm_overwrite") == "on":
        with db.cursor() as cur:
            run_count = cur.execute(
                "SELECT COUNT(*) AS c FROM runs WHERE experiment_id = ?", (name,)
            ).fetchone()["c"]
        note = (
            f" It already has {run_count} recorded run(s) that would stay attached "
            "and be aggregated with whatever this form saves."
            if run_count
            else ""
        )
        field_errors["name"] = (
            f"an experiment named '{name}' already exists.{note} Check 'overwrite' "
            "below to replace it, or pick a different name."
        )

    arms, models = _parse_arms_and_models(form)

    data = {
        "experiment": name,
        "question": form.get("question", "").strip() or None,
        "task": {
            "id": form.get("task_id", "").strip(),
            "success_check": form.get("success_check", "").strip(),
            "timeout_s": timeout_s,
        },
        "models": models,
        "arms": arms,
        "repeats": repeats,
    }

    # Defect 20/22: don't attempt to save (and don't 500) on a bad int or an
    # already-existing name -- re-render the form with the user's input
    # intact and the specific problem called out.
    if field_errors:
        return _new_experiment_response(request, snapshot, field_errors, [])

    try:
        experiment = store.save_experiment(data)
    except store.InvalidExperimentName as e:
        field_errors["name"] = str(e)
        return _new_experiment_response(request, snapshot, field_errors, [])
    except ValidationError as e:
        field_errors, general_errors = _split_validation_errors(e)
        return _new_experiment_response(request, snapshot, field_errors, general_errors)

    return _redirect(f"/experiments/{experiment.experiment}", ok="experiment created")


# ---------------------------------------------------------------------------
# Experiment detail
# ---------------------------------------------------------------------------


def _arm_runs(cur, experiment_name: str, arm_id: str) -> list[dict]:
    rows = cur.execute(
        "SELECT id, repeat_idx, started_at, ended_at, wall_clock_s, task_success "
        "FROM runs WHERE arm_id = ? ORDER BY repeat_idx",
        (runs.arm_row_id(experiment_name, arm_id),),
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/experiments/{name}", response_class=HTMLResponse)
def experiment_detail(request: Request, name: str):
    path, experiment, err = _load_experiment_or_404(request, name)
    if err:
        return err

    with db.cursor() as cur:
        arms_data = [
            {"arm": arm, "runs": _arm_runs(cur, name, arm.id)} for arm in experiment.arms
        ]
    total_runs = sum(len(a["runs"]) for a in arms_data)

    return templates.TemplateResponse(
        request,
        "experiment.html",
        _ctx(
            request,
            experiment=experiment,
            experiment_path=path,
            arms_data=arms_data,
            total_runs=total_runs,
            managed=store.is_managed(path),
        ),
    )


@app.get("/experiments/{name}/yaml", response_class=HTMLResponse)
def experiment_yaml(request: Request, name: str):
    """Raw-YAML view (item 2: "no edit, no YAML view, no delete" -- an
    experiment was write-once through the form; any change meant
    hand-editing a file the UI never showed you). `store.read_raw` was
    already there, unused, clearly meant as this route's hook."""
    path, experiment, err = _load_experiment_or_404(request, name)
    if err:
        return err
    raw = store.read_raw(name)
    return templates.TemplateResponse(
        request,
        "experiment_yaml.html",
        _ctx(request, experiment=experiment, raw=raw, managed=store.is_managed(path)),
    )


@app.get("/experiments/{name}/edit", response_class=HTMLResponse)
def edit_experiment_form(request: Request, name: str):
    path, experiment, err = _load_experiment_or_404(request, name)
    if err:
        return err
    if not store.is_managed(path):
        # Editing (like deleting) is refused for anything discovery found
        # outside EXPERIMENTS_DIR -- see store.is_managed's docstring. The
        # experiment is still fully usable (view YAML, start proxy/runs,
        # compare); only the dashboard's own write path is off-limits, so
        # a repo-committed experiments/*.yaml can't be silently rewritten
        # from a web form.
        return _error_page(
            request,
            400,
            f"'{name}' is defined at {path}, outside the dashboard's own "
            "experiments directory, so it can't be edited here -- edit the "
            "file directly.",
        )
    return templates.TemplateResponse(
        request,
        "new_experiment.html",
        _ctx(
            request,
            form_data=_experiment_to_form_snapshot(experiment),
            field_errors={},
            general_errors=[],
            edit_name=name,
        ),
    )


@app.post("/experiments/{name}/edit")
async def edit_experiment(request: Request, name: str):
    path, _experiment, err = _load_experiment_or_404(request, name)
    if err:
        return err
    if not store.is_managed(path):
        return _error_page(
            request,
            400,
            f"'{name}' is defined at {path}, outside the dashboard's own "
            "experiments directory, so it can't be edited here -- edit the "
            "file directly.",
        )

    form = await request.form()
    snapshot = _form_snapshot(form)
    snapshot["name"] = name  # the name field is read-only in edit mode
    field_errors: dict[str, str] = {}

    timeout_s = _parse_int_field(form, "timeout_s", 1800, field_errors, "timeout (seconds)")
    repeats = _parse_int_field(form, "repeats", 3, field_errors, "repeats per arm")
    arms, models = _parse_arms_and_models(form)

    data = {
        "experiment": name,
        "question": form.get("question", "").strip() or None,
        "task": {
            "id": form.get("task_id", "").strip(),
            "success_check": form.get("success_check", "").strip(),
            "timeout_s": timeout_s,
        },
        "models": models,
        "arms": arms,
        "repeats": repeats,
    }

    if field_errors:
        return _new_experiment_response(request, snapshot, field_errors, [], edit_name=name)

    try:
        store.save_experiment(data)
    except ValidationError as e:
        field_errors, general_errors = _split_validation_errors(e)
        return _new_experiment_response(
            request, snapshot, field_errors, general_errors, edit_name=name
        )

    return _redirect(f"/experiments/{name}", ok="experiment updated")


@app.post("/experiments/{name}/delete")
def delete_experiment(request: Request, name: str):
    """Delete `name`'s definition file. Mirrors defect 21's house style:
    name the run count in the confirmation (experiment.html's JS confirm())
    so the consequence is visible before the user commits, since it's the
    same shape of risk -- the runs recorded against this experiment id are
    *not* deleted (there is no cascade; see ys/runs.py), they just become
    unreachable from the dashboard (which resolves `/experiments/{name}`
    via this same file) until a same-named experiment exists again."""
    path, _experiment, err = _load_experiment_or_404(request, name)
    if err:
        return err
    if not store.is_managed(path):
        return _redirect(
            "/",
            error=(
                f"'{name}' is defined at {path}, outside the dashboard's own "
                "experiments directory, so it can't be deleted here -- remove "
                "the file directly."
            ),
        )
    store.delete_experiment_file(name)
    return _redirect("/", ok=f"deleted experiment '{name}' (recorded runs were not deleted)")


@app.get("/experiments/{name}/compare", response_class=HTMLResponse)
def experiment_compare(request: Request, name: str):
    path, experiment, err = _load_experiment_or_404(request, name)
    if err:
        return err

    with db.cursor() as cur:
        try:
            comparison = render.compare_experiment(cur, experiment)
        except render.CompareError as e:
            return _redirect(f"/experiments/{name}", error=str(e))
        # Item 4: this used to be `HTMLResponse(render.render_html(...))` --
        # a standalone document with no shell and no way back. Embed the
        # report fragment inside the app's own page instead of forking the
        # renderer; render.render_html's `standalone=False` seam exists
        # for exactly this (see its docstring).
        report_html = render.render_html(comparison, cur, standalone=False)

    return templates.TemplateResponse(
        request,
        "compare.html",
        _ctx(request, experiment=experiment, report_html=report_html),
    )


# ---------------------------------------------------------------------------
# Proxy control (scoped to one experiment's models)
# ---------------------------------------------------------------------------


@app.post("/experiments/{name}/proxy/up")
def start_proxy(request: Request, name: str):
    # Defect 23: there was no existence check here at all -- a nonexistent
    # experiment reported whatever the proxy layer complained about first
    # (e.g. "LITELLM_MASTER_KEY is not set", unrelated to the real problem)
    # and redirected to a detail page that then 500s on the same unguarded
    # `store.experiment_path` call defect 19 covers.
    path, _experiment, err = _load_experiment_or_404(request, name)
    if err:
        return err
    try:
        proxy.proxy_up([path])
    except proxy.ProxyError as e:
        return _redirect(f"/experiments/{name}", error=str(e))
    return _redirect(f"/experiments/{name}", ok="proxy started")


@app.post("/proxy/down")
async def stop_proxy(request: Request):
    form = await request.form()
    back = form.get("back") or "/"
    message = proxy.proxy_down()
    return _redirect(back, ok=message)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


@app.post("/experiments/{name}/runs/start")
async def start_run(request: Request, name: str):
    form = await request.form()
    arm_id = form.get("arm_id", "")
    force = form.get("force") == "on"

    path, experiment, err = _load_experiment_or_404(request, name)
    if err:
        return err
    with open(path) as f:
        config_yaml = f.read()

    try:
        result = runs.begin_run(experiment, config_yaml, arm_id, force=force)
    except runs.ArmNotFound as e:
        return _redirect(f"/experiments/{name}", error=str(e))
    except state.RunAlreadyActive as e:
        return _redirect(f"/experiments/{name}", error=str(e))
    except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
        return _redirect(f"/experiments/{name}", error=_write_failed_message(e))

    return _redirect(
        f"/experiments/{name}", ok=f"started run {result.run_id} (repeat {result.repeat_idx})"
    )


@app.post("/runs/end")
async def end_run(request: Request):
    form = await request.form()
    manual_score_raw = form.get("manual_score", "").strip()
    manual_score = float(manual_score_raw) if manual_score_raw else None

    try:
        result = runs.finish_run(manual_score=manual_score)
    except runs.NoActiveRun:
        return _redirect("/", error="no active run")
    except runs.ActiveRunMissingDbRow as e:
        return _redirect("/", error=str(e))
    except runs.NoSuccessCheck as e:
        return _redirect("/", error=f"{e} (set a manual score instead)")
    except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
        return _redirect("/", error=_write_failed_message(e))

    dest = f"/experiments/{result.experiment_name}"
    verdict = "SUCCESS" if result.task_success else "FAIL"
    return _redirect(dest, ok=f"run {result.run_id} finished: {verdict}")


@app.post("/runs/{run_id}/delete")
def delete_run(run_id: str):
    try:
        result = runs.delete_run(run_id)
    except runs.RunNotFound as e:
        return _redirect("/", error=str(e))
    except runs.CannotDeleteActiveRun as e:
        return _redirect(f"/runs/{run_id}", error=str(e))
    except (sqlite3.OperationalError, sqlite3.IntegrityError) as e:
        return _redirect(f"/runs/{run_id}", error=_write_failed_message(e))

    return _redirect(
        f"/experiments/{result.experiment_name}", ok=f"deleted run {result.run_id}"
    )


# ---------------------------------------------------------------------------
# Run detail (raw request log for one run)
# ---------------------------------------------------------------------------


def _turn_chart_svg(main_requests: list, width: int = 640, height: int = 80) -> str:
    """Per-turn context-token chart for one run's main conversation thread
    (item 5's "no per-turn chart"). Plain hand-rolled inline SVG -- the
    same weight as ys/render.py's own `_sparkline_svg` for the static
    report -- rather than pulling in a charting library for one polyline.
    Kept as its own small function here instead of importing render.py's
    private helper across the module boundary."""
    series = [
        (r.get("input_tokens") or 0) + (r.get("cache_creation") or 0) + (r.get("cache_read") or 0)
        for r in main_requests
    ]
    if len(series) < 2:
        return ""
    lo, hi = min(series), max(series)
    span = (hi - lo) or 1
    step = width / (len(series) - 1)
    points = " ".join(
        f"{i * step:.1f},{height - ((v - lo) / span) * height:.1f}" for i, v in enumerate(series)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="context tokens per turn">'
        f'<polyline fill="none" stroke="#4f7cff" stroke-width="2" points="{points}" /></svg>'
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str):
    with db.cursor() as cur:
        run_row = cur.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if run_row is None:
            return _redirect("/", error=f"no such run '{run_id}'")
        requests_rows = cur.execute(
            "SELECT * FROM requests WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        tool_call_rows = cur.execute(
            "SELECT * FROM tool_calls WHERE run_id = ?", (run_id,)
        ).fetchall()
        arm_row = cur.execute(
            "SELECT factors_json FROM arms WHERE id = ?", (run_row["arm_id"],)
        ).fetchone()
        metrics_dict = {}
        chart_svg = ""
        try:
            from ys import metrics as metrics_mod

            metrics_dict = metrics_mod.compute_run_metrics(cur, run_id)
            chart_svg = _turn_chart_svg(metrics_mod._main_requests(cur, run_id))
        except ImportError:
            pass

    factors = json.loads(arm_row["factors_json"]) if arm_row else {}

    return templates.TemplateResponse(
        request,
        "run_detail.html",
        _ctx(
            request,
            run=dict(run_row),
            requests=[dict(r) for r in requests_rows],
            tool_calls=[dict(r) for r in tool_call_rows],
            metrics=metrics_dict,
            factors=factors,
            chart_svg=chart_svg,
        ),
    )


@app.get("/runs/{run_id}/live")
def run_live(run_id: str):
    """JSON polled by base.html's active-run banner and run_detail.html's
    own script (item 3: "nothing updates during a live run" -- the
    dashboard was a static page during the one phase where the user is
    watching). Cheap on purpose: no charting/aggregation beyond what
    token_metrics/turn_metrics already compute, queried fresh each poll --
    this is meant to be hit every few seconds while a run is active, not a
    replacement for the full run_detail page.
    """
    with db.cursor() as cur:
        run_row = cur.execute(
            "SELECT ended_at FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if run_row is None:
            return JSONResponse({"error": f"no such run '{run_id}'"}, status_code=404)
        requests_rows = cur.execute(
            "SELECT seq, model, input_tokens, cache_creation, cache_read, output_tokens, "
            "response_cost, transition FROM requests WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        tokens = metrics.token_metrics(cur, run_id)
        turns = metrics.turn_metrics(cur, run_id)

    return {
        "ended": run_row["ended_at"] is not None,
        "request_count": len(requests_rows),
        "turns": turns["turns"],
        "cost_usd": tokens["cost_usd"],
        "billable_tokens": tokens["billable_tokens"],
        "requests": [dict(r) for r in requests_rows],
    }
