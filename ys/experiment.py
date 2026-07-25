from typing import Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator


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
