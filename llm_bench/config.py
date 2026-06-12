# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""LLM Benchmark configuration loader.

Loads environment variables from ``.env`` and exposes typed settings
for downstream runners.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from loguru import logger


@dataclass(frozen=True)
class BenchConfig:
    """Immutable configuration container for benchmark execution.

    Attributes:
        base_url: OpenAI-compatible API endpoint.
        api_key: Authentication key for the endpoint.
        model: Model identifier passed to the completions endpoint.
    """

    base_url: str
    api_key: str
    model: str


def resolve_default_model(base_url: str, api_key: str) -> str:
    """Fetch the first available model from ``/v1/models``.

    Args:
        base_url: OpenAI-compatible API base URL.
        api_key: API key for authentication.

    Returns:
        The ``id`` of the first model returned by the endpoint.

    Raises:
        RuntimeError: If the request fails or no models are available.
    """
    url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        msg = f"Failed to fetch model list from {url}: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc

    payload: dict[str, Any] = response.json()
    models: list[dict[str, Any]] = payload.get("data", [])
    if not models:
        msg = f"No models available at {url}"
        logger.error(msg)
        raise RuntimeError(msg)

    model_id = str(models[0].get("id", ""))
    if not model_id:
        msg = "First model entry has no 'id' field"
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Resolved default model from {}: {}", url, model_id)
    return model_id


def load_config(dotenv_path: str | Path | None = None) -> BenchConfig:
    """Load benchmark configuration from environment.

    Reads ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``, and ``OPENAI_MODEL``
    from the environment (optionally via a ``.env`` file). When
    ``OPENAI_MODEL`` is not set, the first model from ``/v1/models`` is
    used as the default.

    Args:
        dotenv_path: Explicit path to the ``.env`` file. If ``None``,
            ``python-dotenv`` searches the current working directory.

    Returns:
        A frozen :class:`BenchConfig` instance.

    Raises:
        RuntimeError: If required variables are missing or model
            discovery fails.
    """
    if dotenv_path is not None:
        load_dotenv(dotenv_path=Path(dotenv_path))
        logger.debug("Loaded .env from {}", dotenv_path)
    else:
        load_dotenv()
        logger.debug("Loaded .env from working directory")

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")

    missing: list[str] = []
    if not base_url:
        missing.append("OPENAI_BASE_URL")
    if not api_key:
        missing.append("OPENAI_API_KEY")

    if missing:
        msg = f"Missing environment variables: {', '.join(missing)}"
        logger.error(msg)
        raise RuntimeError(msg)

    assert base_url is not None
    assert api_key is not None

    base_url = base_url.strip()
    api_key = api_key.strip().strip('"').strip("'")

    if not model:
        model = resolve_default_model(base_url, api_key)

    logger.info(
        "Configuration loaded: base_url={} model={}",
        base_url,
        model.strip(),
    )

    return BenchConfig(
        base_url=base_url,
        api_key=api_key,
        model=model.strip(),
    )
