from pathlib import Path
from pydantic import BaseModel
import tomllib
import os

from platformdirs import user_config_dir, user_data_dir


APP_NAME = "odyssey"


class Settings(BaseModel):
    ollama_host: str = "http://localhost:11434"
    fast_model: str = "qwen2.5:14b"
    deep_model: str = "qwen3.6:35b-a3b-coding-nvfp4"
    embedding_model: str = "qwen2.5:14b"
    data_dir: str = ""
    config_dir: str = ""


def get_data_dir() -> Path:
    return Path(
        os.environ.get("ODYSSEY_DATA_DIR")
        or user_data_dir(APP_NAME, ensure_exists=True)
    )


def get_config_dir() -> Path:
    return Path(
        os.environ.get("ODYSSEY_CONFIG_DIR")
        or user_config_dir(APP_NAME, ensure_exists=True)
    )


def load_settings() -> Settings:
    config_dir = get_config_dir()
    config_file = config_dir / "config.toml"
    overrides = {}
    if config_file.exists():
        with open(config_file, "rb") as f:
            overrides = tomllib.load(f)
    base = Settings(
        data_dir=str(get_data_dir()),
        config_dir=str(config_dir),
    )
    return base.model_copy(update=overrides)
