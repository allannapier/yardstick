import itertools
import os
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator


def resolve_model_key(model: Optional[str], table: dict) -> Optional[str]:
    """Match a request/run's recorded `model` string against a dict keyed
    the way `Experiment.models`/`.pricing`/`.billable_weights` are: by the
    experiment's `factors.model` *value* (e.g. `claude-sonnet-5`), not
    necessarily what ends up stored on the row. LiteLLM often records a
    provider-prefixed id (`anthropic/claude-sonnet-5`) while the YAML key is
    the bare factor value -- see `experiments/interactive-sonnet.yaml`. Tries
    the exact string first, then the part after the first `/`. Returns the
    matching key (suitable for `table[key]`), or None if neither matches."""
    if not model or not table:
        return None
    if model in table:
        return model
    if "/" in model:
        stripped = model.split("/", 1)[1]
        if stripped in table:
            return stripped
    return None


class Task(BaseModel):
    id: str
    # `repo`/`ref`/`prompt_file` are declared-but-unconsumed today (findings
    # 15-18 in IMPROVEMENTS.md): nothing in this rig reads them yet. They
    # are *not* dead config to delete -- they are the designated hooks for
    # two queued features: `prompt_file` for feature 1 (unattended runs,
    # `ys run --exp E --arm A --repeats N` driving the agent non-
    # interactively from this file instead of a human), and `repo`/`ref`
    # for feature 2 (workspace isolation -- a per-run git worktree/clone
    # from `repo`@`ref` so every repeat starts from an identical tree).
    # Until those land, all three are validated (see `_ref_requires_repo`
    # below and `ys/cli.py`'s `start()`, which checks `prompt_file` actually
    # exists on disk) but otherwise inert -- validating them now means a
    # typo'd path fails loudly today instead of silently once the feature
    # that reads it finally ships.
    repo: Optional[str] = None
    ref: Optional[str] = None
    prompt_file: Optional[str] = None
    success_check: str
    timeout_s: int = 1800

    @model_validator(mode="after")
    def _ref_requires_repo(self):
        # Schema-only check (no I/O, so it runs at YAML-load time for every
        # command, not just `ys start`): a `ref` with no `repo` to check it
        # out from can never mean anything, today or once feature 2 reads
        # it. Existence of `repo`/`prompt_file` on disk is checked in
        # `ys/cli.py`'s `start()` instead -- that needs the filesystem, and
        # "fail loudly at ys start" is the finding's own suggested fix.
        if self.ref and not self.repo:
            raise ValueError("task.ref is set but task.repo is not -- a ref needs a repo to check it out from")
        return self


def expand_factors(factors: dict[str, list[str]]) -> list[dict[str, str]]:
    """Cartesian product of a declared `factors:` space, e.g.
    `{"agent": ["claude-code"], "model": ["a", "b"]}` ->
    `[{"agent": "claude-code", "model": "a"}, {"agent": "claude-code", "model": "b"}]`.

    Finding 15-18: `factors:` had no consumer at all -- no validation
    against the arms, and no "generate the combinations" helper, which the
    finding calls the obvious reason to declare a factor space rather than
    just writing each arm's `factors:` dict by hand. This is that helper;
    `Experiment.factor_combinations()` is the instance-method form.
    Returns `[]` for an undeclared (default `{}`) factor space -- there's
    no space to expand."""
    if not factors:
        return []
    keys = list(factors.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(factors[k] for k in keys))]


def validate_task_paths(task: "Task") -> list[str]:
    """Filesystem checks for `task.prompt_file`/`task.repo` -- deliberately
    *not* pydantic validators, since those run on every YAML load
    (`ys compare`/`ys report` load the same file and have no need for
    `prompt_file` to exist on a machine that isn't using feature 1 yet).
    Called explicitly from `ys/cli.py`'s `start()` instead, per finding
    15-18's suggested fix: "a prompt_file that doesn't exist should fail
    loudly at ys start, not silently at run time". Returns a list of
    human-readable problems (empty if none) rather than raising, so the
    caller decides how loudly to fail.

    `task.repo` is only checked when it looks like a local filesystem path
    (no `://` scheme, no `user@host:` scp-style remote syntax) -- most real
    usage will be a remote git URL, and confirming a remote repo/ref
    actually exists needs a network call (`git ls-remote`) that belongs to
    feature 2's own implementation, not this stopgap.
    """
    problems = []
    if task.prompt_file and not os.path.isfile(task.prompt_file):
        problems.append(f"task.prompt_file '{task.prompt_file}' does not exist")
    if task.repo and "://" not in task.repo and "@" not in task.repo and not os.path.exists(task.repo):
        problems.append(f"task.repo '{task.repo}' looks like a local path but does not exist")
    return problems


class Arm(BaseModel):
    id: str
    factors: dict[str, str]
    notes: Optional[str] = None
    baseline: bool = False


# Every metric name `ys compare`/`ys report` (ys/render.py) can be told to
# display via `metrics.primary`/`.secondary`/`.derived`, plus the two names
# handled specially (always shown, not part of a per-run mean/spread):
# `cost_per_success` (an aggregate-level ratio, not a per-run metric) and
# `tokens_per_turn` (finding 15-18: previously named in the example YAMLs'
# `derived:` block but never computed at all -- now a real per-run metric,
# `ys/metrics.py`'s `_EFFICIENCY_METRICS` list plus that one aggregate-only
# extra). Kept here as a plain literal rather than importing ys.metrics --
# ys/metrics.py already imports `resolve_model_key` from this module, and
# importing back would be a cycle. tests/test_experiment.py asserts this
# set can't silently drift from ys.metrics's real computed keys.
VALID_METRIC_NAMES = frozenset(
    {
        "billable_tokens",
        "cost_usd",
        "context_high_water",
        "context_growth_rate",
        "cache_read_ratio",
        "overhead_tokens_per_turn",
        "fixed_overhead_tokens",
        "overhead_share",
        "turns",
        "tool_calls",
        "tool_calls_per_turn",
        "tool_error_rate",
        "unique_tool_calls",
        "redundant_tool_calls",
        "redundancy_rate",
        "read_amplification",
        "compaction_events",
        "tokens_dropped",
        "turns_to_recompaction",
        "post_compaction_regrowth",
        "background_requests",
        "background_tokens",
        "wall_clock_s",
        "active_s",
        "tokens_per_turn",
        "cost_per_success",
    }
)

# `metrics.gate` is a string in the YAML so a future gate can be added
# without changing the schema shape -- but a metric name is untrusted user
# input (finding 15-18), so it's validated against this registry rather
# than passed straight to `aggregate_run_metrics` unchecked. Only
# task_success exists today: it's the only boolean pass/fail signal this
# rig currently computes (see ys/metrics.py's `outcome_metrics`).
VALID_GATE_NAMES = frozenset({"task_success"})


class Metrics(BaseModel):
    gate: str = "task_success"
    primary: list[str] = []
    secondary: list[str] = []
    derived: list[str] = []

    @field_validator("gate")
    @classmethod
    def _valid_gate(cls, v):
        if v not in VALID_GATE_NAMES:
            raise ValueError(
                f"unknown metrics.gate '{v}' -- valid options: {', '.join(sorted(VALID_GATE_NAMES))}"
            )
        return v

    @field_validator("primary", "secondary", "derived")
    @classmethod
    def _valid_metric_names(cls, v, info):
        # A misspelled/unknown metric name here used to just silently
        # display nothing (render.py hardcoded its own lists and ignored
        # this block entirely) -- finding 15-18 is explicit that untrusted
        # config like this must fail loudly instead, naming the valid
        # options rather than leaving the user to guess.
        unknown = [name for name in v if name not in VALID_METRIC_NAMES]
        if unknown:
            raise ValueError(
                f"unknown metric name(s) in metrics.{info.field_name}: {unknown} -- "
                f"valid options: {', '.join(sorted(VALID_METRIC_NAMES))}"
            )
        return v


class ModelPricing(BaseModel):
    """Declared USD-per-million-token prices for one model, used to compute
    `cost_usd` when LiteLLM's own cost map has no entry for it (finding 9 in
    IMPROVEMENTS.md) -- LiteLLM silently returns 0.0 in that case, which is
    verified true for `claude-sonnet-5` as configured in
    `experiments/interactive-sonnet.yaml`, and true in general for any
    custom/self-hosted deployment name. Every field is optional: a partial
    declaration (e.g. no cache pricing) still prices input/output tokens,
    it just can't price the token categories it leaves unset."""
    input_per_mtok: Optional[float] = None
    output_per_mtok: Optional[float] = None
    cache_write_per_mtok: Optional[float] = None
    cache_read_per_mtok: Optional[float] = None


class BillableWeights(BaseModel):
    """Per-model weights for `billable_tokens` (finding 10): a
    pricing-*weighted proxy* for spend, not a token count -- it exists to
    approximate relative cost across arms without needing real prices.
    Defaults are Anthropic's cache economics relative to a plain input
    token: a 5-minute cache write costs ~1.25x (Anthropic bills cache
    writes at a premium, not at parity -- the previous hardcoded formula
    used 1.0, which was simply wrong), a cache read ~0.1x (a ~90%
    discount). These defaults are meaningless for a provider with different
    cache economics; declare this model's own weights instead of trusting
    them for anything but Anthropic."""
    input: float = 1.0
    output: float = 1.0
    cache_creation: float = 1.25
    cache_read: float = 0.1


class Experiment(BaseModel):
    experiment: str
    question: Optional[str] = None
    task: Task
    factors: dict[str, list[str]] = {}
    # Maps a `factors.model` value to the litellm_params `ys proxy up` should
    # generate for it in the proxy's model_list (e.g. mock_response for a
    # free test double, or {model: anthropic/..., api_key: os.environ/...}
    # for a real deployment). Not in the original spec's YAML shape -- added
    # because `ys proxy up` needs a concrete source for this and the spec
    # didn't define one. A model factor value with no entry here falls back
    # to the convention `anthropic/<value>` + os.environ/ANTHROPIC_API_KEY.
    models: dict[str, dict] = {}
    # Keyed exactly like `models` above (a `factors.model` value) rather than
    # a parallel mechanism -- these are the two other things the experiment
    # config needs to say about a model besides how to route to it: what it
    # costs when LiteLLM can't price it (finding 9), and how to weight its
    # cache tokens for `billable_tokens` (finding 10). A model with no entry
    # here simply gets no declared-cost fallback / the Anthropic-shaped
    # BillableWeights() default, respectively.
    pricing: dict[str, ModelPricing] = {}
    billable_weights: dict[str, BillableWeights] = {}
    arms: list[Arm]
    repeats: int = 3
    metrics: Metrics = Metrics()

    @field_validator("arms")
    @classmethod
    def _non_empty_arms(cls, v):
        if not v:
            raise ValueError("experiment must declare at least one arm")
        ids = [a.id for a in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate arm ids: {ids}")
        return v

    @model_validator(mode="after")
    def _exactly_one_baseline(self):
        baselines = [a.id for a in self.arms if a.baseline]
        if len(baselines) > 1:
            raise ValueError(f"more than one baseline arm declared: {baselines}")
        return self

    @model_validator(mode="after")
    def _arms_within_declared_factor_space(self):
        # Finding 15-18: `factors:` declares the space arms are supposed to
        # be drawn from, but nothing checked an arm actually stayed inside
        # it -- an arm could reference a factor key nothing declared, or a
        # value (a typo'd model id, say) not in that key's declared list,
        # and the only symptom would be a silently-never-matching entry in
        # `models:`/`pricing:`/`billable_weights:` far downstream (or, for
        # `model`, the proxy's catch-all quietly serving an unregistered
        # model with none of its declared params). Skipped entirely when
        # `factors:` itself isn't declared (the default, `{}`) -- that's an
        # experiment not using the factor-space feature at all, not one
        # with an empty space every arm violates.
        if not self.factors:
            return self
        for arm in self.arms:
            for key, value in arm.factors.items():
                if key not in self.factors:
                    raise ValueError(
                        f"arm '{arm.id}' has factor key '{key}', which is not declared in "
                        f"factors: (declared keys: {sorted(self.factors)})"
                    )
                if value not in self.factors[key]:
                    raise ValueError(
                        f"arm '{arm.id}' has factors.{key}='{value}', which is not among the "
                        f"declared values for '{key}': {self.factors[key]}"
                    )
        return self

    def factor_combinations(self) -> list[dict[str, str]]:
        """The cartesian product of this experiment's declared `factors:`
        space -- see `expand_factors`. An authoring aid (and a building
        block for a future auto-generate-arms command), not something any
        command calls today."""
        return expand_factors(self.factors)

    def get_arm(self, arm_id: str) -> Arm:
        for a in self.arms:
            if a.id == arm_id:
                return a
        raise KeyError(f"no such arm '{arm_id}' in experiment '{self.experiment}'")

    def baseline_arm(self) -> Optional[Arm]:
        for a in self.arms:
            if a.baseline:
                return a
        return None


def load_experiment(path: str) -> Experiment:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Experiment.model_validate(raw)


def dump_raw_yaml(path: str) -> str:
    with open(path) as f:
        return f.read()
