import pytest
from pydantic import ValidationError

from ys import metrics as metrics_module
from ys.experiment import (
    VALID_GATE_NAMES,
    VALID_METRIC_NAMES,
    BillableWeights,
    Experiment,
    expand_factors,
    load_experiment,
    resolve_model_key,
    validate_task_paths,
)

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


# --- findings 15-18: metrics.gate / primary / secondary / derived ----------


def test_valid_metric_names_matches_metrics_module_computed_keys():
    """VALID_METRIC_NAMES is a plain literal (not an import of ys.metrics --
    that would cycle back into this module, see the comment above it) so it
    can silently drift from what ys/metrics.py actually computes. Pin the
    two together here instead."""
    assert VALID_METRIC_NAMES == set(metrics_module._EFFICIENCY_METRICS) | {"cost_per_success"}


def test_example_experiments_declared_metrics_block_is_valid():
    """Both experiments/*.yaml declare a full metrics: block -- if any name
    in it drifted from what's actually computed, loading either file would
    now fail (finding 15-18 requires exactly that)."""
    load_experiment("experiments/example.yaml")
    load_experiment("experiments/interactive-sonnet.yaml")


def test_unknown_metric_name_in_primary_is_rejected():
    with pytest.raises(ValidationError, match="unknown metric name.*billable_toknes"):
        Experiment.model_validate(_base_kwargs(metrics={"primary": ["billable_toknes"]}))


def test_unknown_metric_name_in_secondary_is_rejected():
    with pytest.raises(ValidationError, match="unknown metric name.*metrics.secondary"):
        Experiment.model_validate(_base_kwargs(metrics={"secondary": ["not_a_real_metric"]}))


def test_unknown_metric_name_error_names_valid_options():
    with pytest.raises(ValidationError) as exc_info:
        Experiment.model_validate(_base_kwargs(metrics={"derived": ["nope"]}))
    assert "turns" in str(exc_info.value)  # a real option, named in the error


def test_unknown_gate_name_is_rejected():
    with pytest.raises(ValidationError, match="unknown metrics.gate"):
        Experiment.model_validate(_base_kwargs(metrics={"gate": "vibes"}))


def test_valid_metric_names_and_gate_names_are_accepted():
    exp = Experiment.model_validate(
        _base_kwargs(metrics={"gate": "task_success", "primary": ["turns"], "derived": ["tokens_per_turn"]})
    )
    assert exp.metrics.primary == ["turns"]
    assert exp.metrics.derived == ["tokens_per_turn"]
    assert exp.metrics.gate in VALID_GATE_NAMES


# --- findings 15-18: factors validated against arms + cartesian product ----


def test_expand_factors_cartesian_product():
    combos = expand_factors({"agent": ["claude-code"], "model": ["a", "b"]})
    assert combos == [
        {"agent": "claude-code", "model": "a"},
        {"agent": "claude-code", "model": "b"},
    ]


def test_expand_factors_empty_when_no_factors_declared():
    assert expand_factors({}) == []


def test_experiment_factor_combinations_matches_expand_factors():
    exp = Experiment.model_validate(
        _base_kwargs(
            factors={"model": ["m1", "m2"]},
            arms=[{"id": "a", "factors": {"model": "m1"}}],
        )
    )
    assert exp.factor_combinations() == [{"model": "m1"}, {"model": "m2"}]


def test_arm_referencing_undeclared_factor_value_is_rejected():
    """Regression for finding 15-18: an arm's model factor value that isn't
    among the declared factors.model list (a typo, or a value nobody ever
    added to the list) used to pass validation silently and only surface
    much later as a request the proxy's catch-all serves with none of its
    declared mock_response/params. Must be rejected at YAML-load time."""
    with pytest.raises(ValidationError, match="not among the declared values"):
        Experiment.model_validate(
            _base_kwargs(
                factors={"model": ["claude-sonnet-5"]},
                arms=[{"id": "a", "factors": {"model": "claude-sonnet-6-typo"}}],
            )
        )


def test_arm_referencing_undeclared_factor_key_is_rejected():
    with pytest.raises(ValidationError, match="not declared in"):
        Experiment.model_validate(
            _base_kwargs(
                factors={"model": ["claude-sonnet-5"]},
                arms=[{"id": "a", "factors": {"harness": "claude-code"}}],
            )
        )


def test_arm_factors_not_validated_when_no_factor_space_declared():
    """An experiment that doesn't declare factors: at all (the default,
    `{}`) isn't using the factor-space feature -- arms aren't checked
    against an empty space that would otherwise reject everything."""
    exp = Experiment.model_validate(
        _base_kwargs(arms=[{"id": "a", "factors": {"model": "anything"}}])
    )
    assert exp.get_arm("a").factors == {"model": "anything"}


def test_arm_factors_matching_declared_space_is_accepted():
    exp = Experiment.model_validate(
        _base_kwargs(
            factors={"agent": ["claude-code"], "model": ["m1"]},
            arms=[{"id": "a", "factors": {"agent": "claude-code", "model": "m1"}}],
        )
    )
    assert exp.get_arm("a").factors == {"agent": "claude-code", "model": "m1"}


# --- findings 15-18: task.repo/ref/prompt_file, validated not deleted ------


def test_task_ref_without_repo_is_rejected():
    with pytest.raises(ValidationError, match="task.ref is set but task.repo is not"):
        Experiment.model_validate(_base_kwargs(task={"id": "t0", "success_check": "true", "ref": "main"}))


def test_task_repo_and_ref_together_is_accepted():
    exp = Experiment.model_validate(
        _base_kwargs(task={"id": "t0", "success_check": "true", "repo": "https://example.com/r.git", "ref": "main"})
    )
    assert exp.task.repo == "https://example.com/r.git"
    assert exp.task.ref == "main"


def test_validate_task_paths_flags_missing_prompt_file(tmp_path):
    exp = Experiment.model_validate(
        _base_kwargs(task={"id": "t0", "success_check": "true", "prompt_file": str(tmp_path / "nope.txt")})
    )
    problems = validate_task_paths(exp.task)
    assert len(problems) == 1
    assert "does not exist" in problems[0]


def test_validate_task_paths_accepts_existing_prompt_file(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("do the thing")
    exp = Experiment.model_validate(
        _base_kwargs(task={"id": "t0", "success_check": "true", "prompt_file": str(prompt)})
    )
    assert validate_task_paths(exp.task) == []


def test_validate_task_paths_flags_missing_local_repo(tmp_path):
    missing_repo = str(tmp_path / "no-such-repo")
    exp = Experiment.model_validate(
        _base_kwargs(task={"id": "t0", "success_check": "true", "repo": missing_repo})
    )
    problems = validate_task_paths(exp.task)
    assert len(problems) == 1
    assert "looks like a local path" in problems[0]


def test_validate_task_paths_skips_remote_looking_repo_urls():
    """A repo that looks like a remote git URL isn't checked against the
    filesystem -- confirming it exists/is reachable needs a network call
    that belongs to feature 2's own implementation, not this stopgap."""
    exp = Experiment.model_validate(
        _base_kwargs(task={"id": "t0", "success_check": "true", "repo": "https://example.com/repo.git"})
    )
    assert validate_task_paths(exp.task) == []


def test_validate_task_paths_empty_when_nothing_declared():
    exp = Experiment.model_validate(_base_kwargs())
    assert validate_task_paths(exp.task) == []
