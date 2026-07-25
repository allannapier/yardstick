"""LiteLLM resolves proxy callbacks as a file path relative to config.yaml's
directory, not a normal Python import. This shim lives next to config.yaml
so LiteLLM can find it, and does a normal import of the real implementation
so ys.collector can freely depend on the rest of the ys package.
"""
from ys.collector import yardstick_logger  # noqa: F401
