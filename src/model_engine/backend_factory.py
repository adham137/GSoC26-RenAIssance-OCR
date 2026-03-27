"""
src/model_engine/backend_factory.py
====================================
Backend Factory — Creates Model Backend Instances.

This module reads configuration and returns the appropriate backend
implementation (local or OpenRouter API).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from src.model_engine.base_backend import BaseModelBackend
from src.model_engine.local_backend import LocalModelBackend
from src.model_engine.model_executor import ModelExecutor
from src.model_engine.openrouter_backend import OpenRouterBackend

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def create_backend(config: dict) -> BaseModelBackend:
    """
    Create a model backend instance based on configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary with the following structure:
        {
            "backend": "local" | "openrouter",
            "local": { ... local config ... },  # required if backend == "local"
            "openrouter": { ... openrouter config ... }  # required if backend == "openrouter"
        }

    Returns
    -------
    BaseModelBackend
        Instantiated backend (LocalModelBackend or OpenRouterBackend).

    Raises
    ------
    ValueError
        If backend type is unknown.
    KeyError
        If required configuration keys are missing.
    EnvironmentError
        If OPENROUTER_API_KEY is not set for openrouter backend.
    """
    # Validate config structure
    if "backend" not in config:
        raise KeyError("Missing required config key: 'backend'")

    backend_type = config["backend"]

    if backend_type == "local":
        if "local" not in config:
            raise KeyError("Missing 'local' configuration for local backend")

        local_config = config["local"]

        # Validate required local config keys
        if "model_path" not in local_config:
            raise KeyError("Missing required local config key: 'model_path'")

        # Create ModelExecutor with local config
        executor = ModelExecutor(
            model_name=local_config.get("model_path", "models/Qwen3.5-0.8B"),
            adapter_path=local_config.get("adapter_path"),
            load_in_4bit=local_config.get("load_in_4bit", True),
            device=local_config.get("device", "cuda"),
        )

        # Load the model
        executor.load_model()

        # Wrap in LocalModelBackend
        backend = LocalModelBackend(executor)
        logger.info("Local backend initialized — model=%s", local_config.get("model_path"))
        return backend

    elif backend_type == "openrouter":
        if "openrouter" not in config:
            raise KeyError("Missing 'openrouter' configuration for openrouter backend")

        openrouter_config = config["openrouter"]

        # Validate required openrouter config keys
        if "api_key" not in openrouter_config:
            raise KeyError("Missing required openrouter config key: 'api_key'")
        if "model" not in openrouter_config:
            raise KeyError("Missing required openrouter config key: 'model'")

        # Check for environment variable resolution
        api_key = openrouter_config["api_key"]
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var_name = api_key[2:-1]
            if os.environ.get(env_var_name) is None:
                raise EnvironmentError(
                    f"{env_var_name} environment variable is not set."
                )

        # Create OpenRouterBackend
        backend = OpenRouterBackend(openrouter_config)
        return backend

    else:
        raise ValueError(
            f"Unknown backend: {backend_type}. Must be 'local' or 'openrouter'."
        )
