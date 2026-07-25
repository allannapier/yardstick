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

    arm_ids = form.getlist("arm_id")
    arm_models = form.getlist("arm_model")
    arm_baselines = set(form.getlist("arm_baseline"))  # values of checked baseline radios
    arm_notes = form.getlist("arm_notes")

    arms = []
    for i, arm_id in enumerate(arm_ids):
        if not arm_id.strip():
            continue
        arms.append(
            {
                "id": arm_id.strip(),
                "factors": {"model": arm_models[i].strip()} if i < len(arm_models) else {},
                "baseline": arm_id.strip() in arm_baselines,
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
        "experiment": form.get("name", "").strip(),
        "question": form.get("question", "").strip() or None,
        "task": {
            "id": form.get("task_id", "").strip(),
            "success_check": form.get("success_check", "").strip(),
            "timeout_s": int(form.get("timeout_s") or 1800),
        },
        "models": models,
        "arms": arms,
        "repeats": int(form.get("repeats") or 3),
    }

    try:
        experiment = store.save_experiment(data)
    except (ValidationError, store.InvalidExperimentName) as e:
        return _redirect("/experiments/new", error=str(e))

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
    path = store.experiment_path(name)
    if not os.path.exists(path):
        return _redirect("/", error=f"no experiment named '{name}'")
    experiment = load_experiment(path)

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
    path = store.experiment_path(name)
    if not os.path.exists(path):
        return _redirect("/", error=f"no experiment named '{name}'")
    experiment = load_experiment(path)

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
def start_proxy(name: str):
    path = store.experiment_path(name)
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

    path = store.experiment_path(name)
    experiment = load_experiment(path)
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

    active_before = state.get_active()

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
