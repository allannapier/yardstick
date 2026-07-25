import os

YARDSTICK_HOME = os.path.expanduser(os.environ.get("YARDSTICK_HOME", "~/.yardstick"))
DB_PATH = os.path.join(YARDSTICK_HOME, "yardstick.db")
ACTIVE_RUN_PATH = os.path.join(YARDSTICK_HOME, "active.json")
PROXY_CONFIG_PATH = os.path.join(YARDSTICK_HOME, "proxy_config.yaml")
PROXY_PID_PATH = os.path.join(YARDSTICK_HOME, "proxy.pid")
PROXY_PORT_PATH = os.path.join(YARDSTICK_HOME, "proxy.port")
PROXY_LOG_PATH = os.path.join(YARDSTICK_HOME, "proxy.log")
EXPERIMENTS_DIR = os.path.join(YARDSTICK_HOME, "experiments")

WEB_PID_PATH = os.path.join(YARDSTICK_HOME, "web.pid")
WEB_PORT_PATH = os.path.join(YARDSTICK_HOME, "web.port")
WEB_LOG_PATH = os.path.join(YARDSTICK_HOME, "web.log")


def ensure_home():
    os.makedirs(YARDSTICK_HOME, exist_ok=True)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
