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
