"""Portkey Cloud backend for `ys proxy up --backend portkey` -- an
alternative to ys/proxy.py's LiteLLM proxy that points the harness at the
caller's own hosted Portkey Cloud account (https://api.portkey.ai) instead
of a locally-run multi-provider router.

The two backends differ in a way that isn't just "a different subprocess to
launch":

  - LiteLLM's proxy is a local process, so ys/collector.py can register an
    in-process Python callback (`litellm.integrations.custom_logger.
    CustomLogger`) that captures token/cost/latency data synchronously, as
    each request completes, and writes it straight into yardstick's own
    database.
  - Portkey Cloud is a remote service the caller doesn't run -- there is no
    equivalent in-process hook. This backend's local process
    (ys/portkey_app.py, run under uvicorn the same way ys/proxy.py runs
    `litellm` as a subprocess) is a thin reverse proxy only: it forwards the
    harness's request to Portkey Cloud, swaps in real Portkey auth, and
    tags the request with the active run id via x-portkey-metadata so it
    can be found again afterward. It does not parse bodies or write to the
    database. The actual per-request data is pulled back out of Portkey's
    own Logs Export API once the run ends -- see ys/portkey_collector.py,
    which `ys end` calls automatically (and `ys proxy pull-logs` calls
    manually, for a retry).

Reuses ys/proxy.py's LITELLM_MASTER_KEY as the *local* proxy's own shared
secret (what the harness sends as its API key to reach this process at
all), deliberately, even though the name says "litellm" -- ys/harness.py
and ys/cli.py's `ys start`/`ys harness point` already read exactly that env
var to point/authenticate a harness at "the proxy on localhost:<port>",
whichever backend is actually listening there. Reusing it here means
neither of those needs to know or care which backend is running; inventing
a second, backend-specific local secret would only add a step to every
existing workflow for no real gain in the trust model (whoever holds either
key can already reach the real upstream API directly).

PORTKEY_API_KEY and PORTKEY_VIRTUAL_KEY are the two pieces of *real*
Portkey auth this backend needs from your work account:

  - PORTKEY_API_KEY: your Portkey account's API key, sent upstream as
    `Authorization: Bearer ...` by ys/portkey_app.py.
  - PORTKEY_VIRTUAL_KEY: which of your account's virtual keys (Portkey's
    name for a saved provider credential) requests should route through.
    This prototype applies one virtual key to every request -- it does not
    yet support routing different experiment arms through different
    virtual keys/providers the way ys/proxy.py's per-model `models:` block
    does for LiteLLM. Fine for a single-provider comparison (e.g. several
    Claude Code arms, all via Anthropic through Portkey); a genuine
    multi-provider comparison needs that per-model mapping added first,
    not guessed at here -- ask whoever admins your work Portkey account
    what virtual key(s) exist before relying on this.
"""
import os

from ys import paths, procutil
from ys.proxy import DEFAULT_PORT, ProxyError

PORTKEY_BASE_URL = "https://api.portkey.ai"


def proxy_up(port: int = DEFAULT_PORT) -> str:
    if not os.environ.get("LITELLM_MASTER_KEY"):
        raise ProxyError(
            "LITELLM_MASTER_KEY is not set. Export a key of your choosing before "
            "starting the proxy, e.g.\n"
            "    export LITELLM_MASTER_KEY=sk-...\n"
            "This is the *local* proxy's own key (reused across both backends -- see "
            "ys/portkey_backend.py's module docstring) -- it's what your harness sends "
            "as its API key, not a real Portkey credential."
        )
    if not os.environ.get("PORTKEY_API_KEY"):
        raise ProxyError(
            "PORTKEY_API_KEY is not set. Export your work Portkey account's API key, e.g.\n"
            "    export PORTKEY_API_KEY=pk-...\n"
            "This is the real credential ys/portkey_app.py forwards requests to Portkey "
            "Cloud with -- distinct from LITELLM_MASTER_KEY above and from "
            "PORTKEY_ADMIN_API_KEY (needed later, by `ys end`/`ys proxy pull-logs`, to "
            "read logs back)."
        )
    if not os.environ.get("PORTKEY_VIRTUAL_KEY"):
        raise ProxyError(
            "PORTKEY_VIRTUAL_KEY is not set. Export the virtual key your work Portkey "
            "account should route requests through, e.g.\n"
            "    export PORTKEY_VIRTUAL_KEY=anthropic-prod\n"
            "Ask whoever admins your work Portkey account which virtual key to use -- "
            "this prototype applies one virtual key to every request (see "
            "ys/portkey_backend.py's module docstring for why)."
        )

    existing = procutil.read_pid(paths.PROXY_PID_PATH)
    if existing and procutil.alive(existing):
        raise ProxyError(
            f"proxy already running (pid {existing}). Run `ys proxy down` first."
        )
    if procutil.port_in_use(port):
        raise ProxyError(
            f"port {port} is already bound by a process ys has no pidfile for -- a proxy "
            "started outside ys, a previous run `ys proxy down` couldn't kill, or an "
            "unrelated service. Free the port, or start with a different --port."
        )

    pid = procutil.launch_detached(
        ["uvicorn", "ys.portkey_app:app", "--host", "0.0.0.0", "--port", str(port)],
        paths.PROXY_LOG_PATH,
        paths.PROXY_PID_PATH,
        paths.PROXY_PORT_PATH,
        port,
    )

    if not procutil.wait_ready(f"http://localhost:{port}/health/readiness"):
        raise ProxyError(
            f"proxy (pid {pid}) did not become ready within 15s. "
            f"Check the log at {paths.PROXY_LOG_PATH}"
        )

    return f"http://localhost:{port}"
