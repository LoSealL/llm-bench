# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""OpenAI-compatible API client with retry and truncation.

Provides a thin wrapper around the official ``openai`` SDK that adds
automatic retry logic and token-based prompt truncation.
"""

import time
from dataclasses import dataclass
from typing import Any

import tiktoken
from loguru import logger
from openai import OpenAI

from llm_bench.config import BenchConfig


@dataclass(frozen=True)
class ChatResponse:
    """Structured response from a chat completion request.

    Attributes:
        content: The assistant's textual response.
        finish_reason: The completion finish reason, if provided.
        valid: ``True`` when *finish_reason* is ``stop``,
            ``tool_calls``, or ``function_call``.
        tool_calls: Native tool calls from ``message.tool_calls`` when
            the request used the ``tools`` parameter, otherwise ``None``.
    """

    content: str
    finish_reason: str | None
    valid: bool
    tool_calls: list[Any] | None = None

    def __bool__(self) -> bool:
        """Return whether the response is valid and non-empty."""
        return self.valid and (bool(self.content) or bool(self.tool_calls))


class LLMClient:
    """Thread-safe OpenAI-compatible chat client.

    Attributes:
        _client: Underlying ``openai.OpenAI`` instance.
        _model: Model name sent in the ``model`` field of every request.
        _tokenizer: ``tiktoken`` encoder used for prompt truncation.
    """

    VALID_FINISH_REASONS = {"stop", "tool_calls", "function_call"}

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
        self._enable_thinking = config.enable_thinking
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
        tools: Any | None = None,
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
            tools: Optional OpenAI-style tool definitions enabling native
                function calling via ``/v1/chat/completions``.

        Returns:
            The assistant's textual response, or an empty string if all
            retries are exhausted.
        """
        return self.chat_with_meta(
            prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            tools=tools,
        ).content

    def chat_with_meta(
        self,
        prompt: str | None = None,
        *,
        messages: Any | None = None,
        max_tokens: int = 128,
        temperature: float = 0.1,
        max_retries: int = 5,
        tools: Any | None = None,
    ) -> ChatResponse:
        """Send a chat request and return content plus finish metadata.

        Args:
            prompt: Raw user message content. Used when ``messages`` is
                not provided.
            messages: Full message list to send directly. Takes precedence
                over ``prompt``.
            max_tokens: Maximum number of *new* tokens to generate.
            temperature: Sampling temperature.
            max_retries: Number of retry attempts on transient failures.
            tools: Optional OpenAI-style tool definitions enabling native
                function calling via ``/v1/chat/completions``. When
                provided, any tool calls are returned in
                :attr:`ChatResponse.tool_calls`.

        Returns:
            A :class:`ChatResponse` with the assistant's text, finish
            reason, validity flag, and optional tool calls.
        """
        if messages is not None:
            text_parts: list[str] = []
            for m in messages:
                content = m.get("content", "")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(str(part.get("text", "")))
            prompt_text = "".join(text_parts)
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
            "tools_count": len(tools) if tools else 0,
        }
        logger.trace("[REQUEST PAYLOAD] {}", payload)
        if tools:
            logger.trace("[TOOLS] {}", tools)

        for attempt in range(1, max_retries + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if tools:
                    request_kwargs["tools"] = tools
                if not self._enable_thinking:
                    request_kwargs["enable_thinking"] = False
                response = self._client.chat.completions.create(**request_kwargs)
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
                content = msg.content or ""
                tool_calls = msg.tool_calls
                is_valid = finish_reason in self.VALID_FINISH_REASONS
                if content:
                    logger.debug("Received response ({} chars)", len(content))
                if tool_calls:
                    logger.debug("Received {} tool call(s)", len(tool_calls))
                if content or tool_calls:
                    return ChatResponse(
                        content=content,
                        finish_reason=finish_reason,
                        valid=is_valid,
                        tool_calls=list(tool_calls) if tool_calls else None,
                    )
                logger.warning("Empty response from model")
                return ChatResponse(
                    content="",
                    finish_reason=finish_reason,
                    valid=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "API call failed (attempt {}/{}): {}", attempt, max_retries, exc
                )
                if attempt == max_retries:
                    logger.error("Max retries reached, giving up: {}", exc)
                    return ChatResponse(
                        content="",
                        finish_reason=None,
                        valid=False,
                    )
                time.sleep(1)
        return ChatResponse(
            content="",
            finish_reason=None,
            valid=False,
        )

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
