import pytest
from pydantic import ValidationError

from ys.experiment import Experiment, load_experiment

EXAMPLE_PATH = "experiments/example.yaml"


def test_loads_example_experiment():
    exp = load_experiment(EXAMPLE_PATH)
    assert exp.experiment == "mock-smoke-01"
    assert exp.repeats == 2
    assert [a.id for a in exp.arms] == ["arm-a", "arm-b"]
    assert exp.baseline_arm().id == "arm-a"
    assert "probe-claude-mock" in exp.models


def test_get_arm_lookup():
    exp = load_experiment(EXAMPLE_PATH)
    assert exp.get_arm("arm-b").notes == "second mock arm, deliberately noisier"
    with pytest.raises(KeyError):
        exp.get_arm("does-not-exist")


def _base_kwargs(**overrides):
    kwargs = dict(
        experiment="t",
        task={"id": "t0", "success_check": "true"},
        arms=[{"id": "a", "factors": {}}],
    )
    kwargs.update(overrides)
    return kwargs


def test_rejects_duplicate_arm_ids():
    with pytest.raises(ValidationError):
        Experiment.model_validate(
            _base_kwargs(
                arms=[
                    {"id": "dup", "factors": {}},
                    {"id": "dup", "factors": {}},
                ]
            )
        )


def test_rejects_multiple_baselines():
    with pytest.raises(ValidationError):
        Experiment.model_validate(
            _base_kwargs(
                arms=[
                    {"id": "a", "factors": {}, "baseline": True},
                    {"id": "b", "factors": {}, "baseline": True},
                ]
            )
        )


def test_rejects_empty_arms():
    with pytest.raises(ValidationError):
        Experiment.model_validate(_base_kwargs(arms=[]))


def test_baseline_arm_none_when_undeclared():
    exp = Experiment.model_validate(_base_kwargs())
    assert exp.baseline_arm() is None
