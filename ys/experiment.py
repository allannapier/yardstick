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
    repo: Optional[str] = None
    ref: Optional[str] = None
    prompt_file: Optional[str] = None
    success_check: str
    timeout_s: int = 1800


class Arm(BaseModel):
    id: str
    factors: dict[str, str]
    notes: Optional[str] = None
    baseline: bool = False


class Metrics(BaseModel):
    gate: str = "task_success"
    primary: list[str] = []
    secondary: list[str] = []
    derived: list[str] = []


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
