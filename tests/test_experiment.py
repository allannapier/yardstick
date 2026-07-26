import pytest
from pydantic import ValidationError

from ys.experiment import BillableWeights, Experiment, load_experiment, resolve_model_key

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


# --- findings 9/10: declared pricing / billable weights ---------------------


def test_pricing_and_billable_weights_default_to_empty():
    exp = Experiment.model_validate(_base_kwargs())
    assert exp.pricing == {}
    assert exp.billable_weights == {}


def test_experiment_parses_declared_pricing_block():
    exp = Experiment.model_validate(
        _base_kwargs(
            pricing={
                "claude-sonnet-5": {
                    "input_per_mtok": 3.0,
                    "output_per_mtok": 15.0,
                    "cache_write_per_mtok": 3.75,
                    "cache_read_per_mtok": 0.3,
                }
            }
        )
    )
    price = exp.pricing["claude-sonnet-5"]
    assert price.input_per_mtok == 3.0
    assert price.output_per_mtok == 15.0
    assert price.cache_write_per_mtok == 3.75
    assert price.cache_read_per_mtok == 0.3


def test_pricing_fields_are_all_optional():
    """A partial declaration (e.g. no cache pricing) is valid -- it just
    can't price the token categories it leaves unset."""
    exp = Experiment.model_validate(
        _base_kwargs(pricing={"m": {"input_per_mtok": 1.0}})
    )
    price = exp.pricing["m"]
    assert price.input_per_mtok == 1.0
    assert price.output_per_mtok is None


def test_billable_weights_default_matches_anthropic_shaped_default():
    """finding 10: the Anthropic-shaped default itself, not just that a
    default exists -- a cache write costs ~1.25x a plain input token, not
    1.0x (the bug this finding fixes), and a cache read ~0.1x."""
    weights = BillableWeights()
    assert weights.input == 1.0
    assert weights.output == 1.0
    assert weights.cache_creation == 1.25
    assert weights.cache_read == 0.1


def test_billable_weights_block_overrides_defaults_partially():
    exp = Experiment.model_validate(
        _base_kwargs(billable_weights={"claude-sonnet-5": {"cache_read": 0.05}})
    )
    weights = exp.billable_weights["claude-sonnet-5"]
    assert weights.cache_read == 0.05
    assert weights.input == 1.0  # untouched fields keep the Anthropic default


# --- resolve_model_key ------------------------------------------------------


def test_resolve_model_key_exact_match():
    assert resolve_model_key("claude-sonnet-5", {"claude-sonnet-5": {}}) == "claude-sonnet-5"


def test_resolve_model_key_strips_provider_prefix():
    """LiteLLM often records a provider-prefixed model id
    (`anthropic/claude-sonnet-5`) while the `pricing:`/`billable_weights:`
    key is the bare `factors.model` value -- see
    experiments/interactive-sonnet.yaml."""
    assert (
        resolve_model_key("anthropic/claude-sonnet-5", {"claude-sonnet-5": {}})
        == "claude-sonnet-5"
    )


def test_resolve_model_key_returns_none_when_no_match():
    assert resolve_model_key("gpt-4", {"claude-sonnet-5": {}}) is None
    assert resolve_model_key(None, {"claude-sonnet-5": {}}) is None
    assert resolve_model_key("claude-sonnet-5", {}) is None
