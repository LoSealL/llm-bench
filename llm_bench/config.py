# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""LLM Benchmark configuration loader.

Loads environment variables from ``.env`` and exposes typed settings
for downstream runners.
"""

import os
from dataclasses import dataclass
from pathlib import Path

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


def load_config(dotenv_path: str | Path | None = None) -> BenchConfig:
    """Load benchmark configuration from environment.

    Reads ``OPENAI_BASE_URL``, ``OPENAI_API_KEY``, and ``OPENAI_MODEL``
    from the environment (optionally via a ``.env`` file).

    Args:
        dotenv_path: Explicit path to the ``.env`` file. If ``None``,
            ``python-dotenv`` searches the current working directory.

    Returns:
        A frozen :class:`BenchConfig` instance.

    Raises:
        RuntimeError: If any required variable is missing.
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
    if not model:
        missing.append("OPENAI_MODEL")

    if missing:
        msg = f"Missing environment variables: {', '.join(missing)}"
        logger.error(msg)
        raise RuntimeError(msg)

    assert base_url is not None
    assert api_key is not None
    assert model is not None

    logger.info(
        "Configuration loaded: base_url={} model={}",
        base_url.strip(),
        model.strip(),
    )

    return BenchConfig(
        base_url=base_url.strip(),
        api_key=api_key.strip().strip('"').strip("'"),
        model=model.strip(),
    )
