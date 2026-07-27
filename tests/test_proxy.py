import yaml

from ys import proxy


def _write_experiment(path, models=None, arms=None):
    content = {
        "experiment": "e1",
        "task": {"id": "t0", "success_check": "true"},
        "models": models or {},
        "arms": arms or [{"id": "a", "factors": {"model": "some-model"}}],
    }
    path.write_text(yaml.safe_dump(content))
    return str(path)


def test_generate_config_uses_explicit_models_block(tmp_path):
    exp_path = _write_experiment(
        tmp_path / "exp.yaml",
        models={"probe-mock": {"model": "anthropic/claude-3-5-sonnet-20241022", "mock_response": "hi"}},
        arms=[{"id": "a", "factors": {"model": "probe-mock"}}],
    )
    config_path = proxy.generate_config([exp_path])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    names = {m["model_name"] for m in config["model_list"]}
    assert "probe-mock" in names
    entry = next(m for m in config["model_list"] if m["model_name"] == "probe-mock")
    assert entry["litellm_params"]["mock_response"] == "hi"


def test_generate_config_fallback_convention(tmp_path):
    exp_path = _write_experiment(
        tmp_path / "exp.yaml",
        models={},
        arms=[{"id": "a", "factors": {"model": "claude-haiku-4-5-20251001"}}],
    )
    config_path = proxy.generate_config([exp_path])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    entry = next(m for m in config["model_list"] if m["model_name"] == "claude-haiku-4-5-20251001")
    assert entry["litellm_params"]["model"] == "anthropic/claude-haiku-4-5-20251001"
    assert entry["litellm_params"]["api_key"] == "os.environ/ANTHROPIC_API_KEY"


def test_generate_config_writes_shim_next_to_config(tmp_path):
    exp_path = _write_experiment(tmp_path / "exp.yaml")
    config_path = proxy.generate_config([exp_path])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    callback = config["litellm_settings"]["callbacks"]
    shim_module = callback.split(".")[0]
    import os

    shim_path = os.path.join(os.path.dirname(config_path), f"{shim_module}.py")
    assert os.path.exists(shim_path)
    with open(shim_path) as f:
        assert "from ys.collector import yardstick_logger" in f.read()


def test_generate_config_raises_when_no_models_resolvable(tmp_path):
    exp_path = _write_experiment(tmp_path / "exp.yaml", models={}, arms=[{"id": "a", "factors": {}}])
    try:
        proxy.generate_config([exp_path])
        assert False, "expected ProxyError"
    except proxy.ProxyError:
        pass


def test_read_port_defaults_when_no_file():
    assert proxy.read_port() == proxy.DEFAULT_PORT
    assert proxy.proxy_status() == (False, None)


def test_generate_config_always_includes_catch_all_entry(tmp_path):
    exp_path = _write_experiment(tmp_path / "exp.yaml")
    config_path = proxy.generate_config([exp_path])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    catch_all = next(m for m in config["model_list"] if m["model_name"] == "*")
    assert catch_all["litellm_params"]["model"] == "anthropic/*"
    assert catch_all["litellm_params"]["api_key"] == "os.environ/ANTHROPIC_API_KEY"


# --- feature 5: provider-agnostic model fallback / catch-all ---------------


def test_generate_config_fallback_does_not_double_prefix_an_already_prefixed_model(tmp_path):
    """Regression test for the pre-feature-5 bug: `_fallback_params` used to
    blindly prepend `anthropic/` to every `factors.model` value with no
    explicit `models:` entry, so an already-provider-prefixed value like
    `openai/gpt-4o` came out as `anthropic/openai/gpt-4o` -- reverting the
    `"/" in model_value` check in `ys.proxy._fallback_params` makes this
    fail."""
    exp_path = _write_experiment(
        tmp_path / "exp.yaml",
        models={},
        arms=[{"id": "a", "factors": {"model": "openai/gpt-4o"}}],
    )
    config_path = proxy.generate_config([exp_path])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    entry = next(m for m in config["model_list"] if m["model_name"] == "openai/gpt-4o")
    assert entry["litellm_params"]["model"] == "openai/gpt-4o"
    assert entry["litellm_params"]["api_key"] == "os.environ/OPENAI_API_KEY"


def test_generate_config_fallback_omits_api_key_for_a_provider_without_a_simple_one(tmp_path):
    """bedrock/vertex_ai use AWS SigV4 creds / a GCP service-account file,
    not a plain bearer token -- `_fallback_params` must not fabricate an
    `api_key` field pointed at a made-up env var name for them."""
    exp_path = _write_experiment(
        tmp_path / "exp.yaml",
        models={},
        arms=[{"id": "a", "factors": {"model": "bedrock/anthropic.claude-3-5-sonnet"}}],
    )
    config_path = proxy.generate_config([exp_path])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    entry = next(m for m in config["model_list"] if m["model_name"] == "bedrock/anthropic.claude-3-5-sonnet")
    assert entry["litellm_params"]["model"] == "bedrock/anthropic.claude-3-5-sonnet"
    assert "api_key" not in entry["litellm_params"]


def test_catch_all_follows_the_single_declared_provider(tmp_path):
    """When every declared model agrees on a provider, the catch-all should
    route background/unregistered traffic to that same provider instead of
    always assuming Anthropic -- a Codex CLI or Aider run's own background
    traffic would carry an OpenAI-shaped model id, and an anthropic/*
    catch-all would send that straight to the wrong API."""
    exp_path = _write_experiment(
        tmp_path / "exp.yaml",
        models={"gpt4-arm": {"model": "openai/gpt-4o", "api_key": "os.environ/OPENAI_API_KEY"}},
        arms=[{"id": "a", "factors": {"model": "gpt4-arm"}}],
    )
    config_path = proxy.generate_config([exp_path])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    catch_all = next(m for m in config["model_list"] if m["model_name"] == "*")
    assert catch_all["litellm_params"]["model"] == "openai/*"
    assert catch_all["litellm_params"]["api_key"] == "os.environ/OPENAI_API_KEY"


def test_catch_all_falls_back_to_anthropic_when_providers_are_mixed(tmp_path):
    """A genuine cross-provider comparison (the tool's own premise) can't
    pick a single catch-all provider -- falls back to the original
    anthropic/* default rather than guessing between the two."""
    exp_path = _write_experiment(
        tmp_path / "exp.yaml",
        models={
            "claude-arm": {"model": "anthropic/claude-sonnet-5", "api_key": "os.environ/ANTHROPIC_API_KEY"},
            "gpt-arm": {"model": "openai/gpt-4o", "api_key": "os.environ/OPENAI_API_KEY"},
        },
        arms=[
            {"id": "a", "factors": {"model": "claude-arm"}},
            {"id": "b", "factors": {"model": "gpt-arm"}},
        ],
    )
    config_path = proxy.generate_config([exp_path])
    with open(config_path) as f:
        config = yaml.safe_load(f)

    catch_all = next(m for m in config["model_list"] if m["model_name"] == "*")
    assert catch_all["litellm_params"]["model"] == "anthropic/*"
    assert catch_all["litellm_params"]["api_key"] == "os.environ/ANTHROPIC_API_KEY"


def test_model_available_returns_none_when_proxy_unreachable():
    # Nothing is listening on this port during a unit test.
    assert proxy.model_available("some-model", 65432, "sk-test") is None


def test_model_available_true_and_false_from_proxy_response(monkeypatch):
    import io
    import json

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return json.dumps(self._payload).encode()

        def __enter__(self):
            return io.BytesIO(self.read())

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        return FakeResponse({"data": [{"id": "claude-sonnet-5"}, {"id": "*"}]})

    monkeypatch.setattr(proxy.urllib.request, "urlopen", fake_urlopen)

    assert proxy.model_available("claude-sonnet-5", 4000, "sk-test") is True
    assert proxy.model_available("claude-haiku-4-5", 4000, "sk-test") is False
