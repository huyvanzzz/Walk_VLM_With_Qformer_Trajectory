import os
from pathlib import Path


def load_env_file(path):
    loaded = {}
    env_path = Path(path)
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded[key.strip()] = value.strip().strip('"').strip("'")
    return loaded


def load_runtime_env(repo_root):
    merged = {}
    for candidate in [Path(repo_root) / ".env", Path(repo_root) / "gptscore" / ".env"]:
        merged.update(load_env_file(candidate))
    return merged


def get_env_value(name, loaded_env=None):
    if name in os.environ:
        return os.environ[name]
    if loaded_env and name in loaded_env:
        return loaded_env[name]
    return None

