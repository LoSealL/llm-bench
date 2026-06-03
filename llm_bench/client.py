# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""OpenAI-compatible API client with retry and truncation.

Provides a thin wrapper around the official ``openai`` SDK that adds
automatic retry logic and token-based prompt truncation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import tiktoken
from openai import OpenAI

if TYPE_CHECKING:
    from llm_bench.config import BenchConfig


class LLMClient:
    """Thread-safe OpenAI-compatible chat client.

    Attributes:
        _client: Underlying ``openai.OpenAI`` instance.
        _model: Model name sent in the ``model`` field of every request.
        _tokenizer: ``tiktoken`` encoder used for prompt truncation.
    """

    def __init__(self, config: BenchConfig) -> None:
        """Initialize the client from a :class:`BenchConfig`.

        Args:
            config: Frozen configuration object.
        """
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self._model = config.model
        self._tokenizer = tiktoken.encoding_for_model(
            "gpt-4o-2024-08-06",
        )

    def chat(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.1,
        max_retries: int = 5,
    ) -> str:
        """Send a single-turn chat request with retries.

        Args:
            prompt: Raw user message content.
            max_tokens: Maximum number of *new* tokens to generate.
            temperature: Sampling temperature.
            max_retries: Number of retry attempts on transient failures.

        Returns:
            The assistant's textual response, or an empty string if all
            retries are exhausted.
        """
        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                return content if content is not None else ""
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries:
                    print(f"Max retries reached: {exc}")
                    return ""
                time.sleep(1)
        return ""

    def truncate_prompt(self, prompt: str, max_length: int) -> str:
        """Truncate a prompt to ``max_length`` tokens using middle-drop.

        Args:
            prompt: Raw prompt text.
            max_length: Maximum token budget.

        Returns:
            Truncated prompt string.
        """
        tokens = self._tokenizer.encode(prompt)
        if len(tokens) <= max_length:
            return prompt
        half = max_length // 2
        truncated = tokens[:half] + tokens[-half:]
        return self._tokenizer.decode(truncated)
