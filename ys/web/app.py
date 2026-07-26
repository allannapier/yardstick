"""Yardstick dashboard: experiment authoring, run start/stop, results
browsing. A thin presentation layer over ys/db.py, ys/runs.py, ys/proxy.py,
and ys/render.py -- it contains no business logic of its own beyond request
parsing and HTML rendering, so it can never drift from what `ys` the CLI
does (see ys/runs.py's docstring for why that split exists).
"""
import os
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from ys import db, proxy, render, runs, state
from ys.experiment import load_experiment
from ys.web import store

app = FastAPI(title="yardstick")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


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
    (`store.experiment_path` raising `InvalidExperimentName`) and a
    well-formed but nonexistent one. Returns `(path, experiment, None)` on
    success or `(None, None, error_response)` on failure -- callers do
    `path, experiment, err = ...; if err: return err`.
    """
    try:
        path = store.experiment_path(name)
    except store.InvalidExperimentName as e:
        return None, None, _error_page(request, 404, str(e))
    if not os.path.exists(path):
        return None, None, _error_page(request, 404, f"no experiment named '{name}'")
    return path, load_experiment(path), None


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


def _new_experiment_response(
    request: Request,
    form_data: dict,
    field_errors: dict,
    general_errors: list,
    status_code: int = 400,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "new_experiment.html",
        _ctx(request, form_data=form_data, field_errors=field_errors, general_errors=general_errors),
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
            summaries.append({"experiment": exp, "run_count": row["c"]})
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

    # arm_seq/arm_baseline are a matched pair from the radio group in
    # new_experiment.html (defect 24): each row gets a stable seq at
    # creation, and the single `arm_baseline` value the browser submits is
    # whichever row's radio was checked. Because there's exactly one
    # `arm_baseline` field name shared by every row, the browser's native
    # radio-group behaviour makes "two arms marked baseline" unreachable
    # from the form -- unlike the old checkboxes, which could both be
    # checked and only failed later as a raw pydantic error (defect 22).
    arm_ids = form.getlist("arm_id")
    arm_models = form.getlist("arm_model")
    arm_seqs = form.getlist("arm_seq")
    baseline_seq = form.get("arm_baseline")
    arm_notes = form.getlist("arm_notes")

    arms = []
    for i, arm_id in enumerate(arm_ids):
        if not arm_id.strip():
            continue
        seq = arm_seqs[i] if i < len(arm_seqs) else None
        arms.append(
            {
                "id": arm_id.strip(),
                "factors": {"model": arm_models[i].strip()} if i < len(arm_models) else {},
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
            models[key.strip()] = {
                "model": "anthropic/claude-3-5-sonnet-20241022",
                "mock_response": value or "mock response",
            }
        else:
            models[key.strip()] = {
                "model": value if value.startswith("anthropic/") else f"anthropic/{value}",
                "api_key": "os.environ/ANTHROPIC_API_KEY",
            }

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

    return templates.TemplateResponse(
        request,
        "experiment.html",
        _ctx(request, experiment=experiment, experiment_path=path, arms_data=arms_data),
    )


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
        content = render.render_html(comparison, cur)

    return HTMLResponse(content)


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

    return _redirect(
        f"/experiments/{result.experiment_name}", ok=f"deleted run {result.run_id}"
    )


# ---------------------------------------------------------------------------
# Run detail (raw request log for one run)
# ---------------------------------------------------------------------------


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
        metrics_dict = {}
        try:
            from ys import metrics

            metrics_dict = metrics.compute_run_metrics(cur, run_id)
        except ImportError:
            pass

    return templates.TemplateResponse(
        request,
        "run_detail.html",
        _ctx(
            request,
            run=dict(run_row),
            requests=[dict(r) for r in requests_rows],
            tool_calls=[dict(r) for r in tool_call_rows],
            metrics=metrics_dict,
        ),
    )
