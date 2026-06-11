# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""OpenAI-compatible API client with retry and truncation.

Provides a thin wrapper around the official ``openai`` SDK that adds
automatic retry logic and token-based prompt truncation.
"""

import time
from typing import Any

import tiktoken
from loguru import logger
from openai import OpenAI

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
        prompt: str | None = None,
        *,
        messages: Any | None = None,
        max_tokens: int = 128,
        temperature: float = 0.1,
        max_retries: int = 5,
    ) -> str:
        """Send a chat request with retries.

        Args:
            prompt: Raw user message content. Used when ``messages`` is
                not provided.
            messages: Full message list to send directly. Takes precedence
                over ``prompt``.
            max_tokens: Maximum number of *new* tokens to generate.
            temperature: Sampling temperature.
            max_retries: Number of retry attempts on transient failures.

        Returns:
            The assistant's textual response, or an empty string if all
            retries are exhausted.
        """
        if messages is not None:
            prompt_text = "".join(m.get("content", "") for m in messages)
        elif prompt is not None:
            prompt_text = prompt
            messages = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        prompt_tokens = len(self._tokenizer.encode(prompt_text))
        logger.trace(
            "[REQUEST] POST {}  model={}  temperature={}  max_tokens={}  "
            "prompt_tokens={}",
            self._client.base_url,
            self._model,
            temperature,
            max_tokens,
            prompt_tokens,
        )
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        logger.trace("[REQUEST PAYLOAD] {}", payload)

        for attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage = response.usage
                finish_reason = response.choices[0].finish_reason
                logger.trace(
                    "[RESPONSE] finish_reason={}  "
                    "prompt_tokens={}  completion_tokens={}  total_tokens={}",
                    finish_reason,
                    usage.prompt_tokens if usage else "N/A",
                    usage.completion_tokens if usage else "N/A",
                    usage.total_tokens if usage else "N/A",
                )

                msg = response.choices[0].message
                content = msg.content
                if content:
                    logger.debug("Received response ({} chars)", len(content))
                    return content
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    logger.debug("Received reasoning_content instead of content")
                    return reasoning
                logger.warning("Empty response from model")
                return ""
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "API call failed (attempt {}/{}): {}", attempt, max_retries, exc
                )
                if attempt == max_retries:
                    logger.error("Max retries reached, giving up: {}", exc)
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
