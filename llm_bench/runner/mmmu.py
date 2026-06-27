# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""MMMU benchmark runner.

Evaluates multimodal understanding on college-level questions across
30 subjects grouped into 6 domains using OpenAI-compatible multimodal
chat APIs. Only single-image questions are evaluated; multi-image samples
are skipped.
"""

import ast
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger

from llm_bench.client import LLMClient
from llm_bench.runners import BaseRunner, _JsonlWriter


class MMMURunner(BaseRunner):
    """Execute the MMMU benchmark suite.

    Evaluates multimodal understanding on college-level questions across
    30 subjects grouped into 6 domains. Only single-image questions are
    evaluated (multi-image samples are skipped).

    Attributes:
        _client: :class:`LLMClient` instance for API calls.
        _output_dir: Directory where predictions are saved.
        _max_tokens: Maximum new tokens for answer generation.
        _temperature: Sampling temperature.
    """

    _SUBJECTS = [
        "Accounting",
        "Agriculture",
        "Architecture_and_Engineering",
        "Art",
        "Art_Theory",
        "Basic_Medical_Science",
        "Biology",
        "Chemistry",
        "Clinical_Medicine",
        "Computer_Science",
        "Design",
        "Diagnostics_and_Laboratory_Medicine",
        "Economics",
        "Electronics",
        "Energy_and_Power",
        "Finance",
        "Geography",
        "History",
        "Literature",
        "Manage",
        "Marketing",
        "Materials",
        "Math",
        "Mechanical_Engineering",
        "Music",
        "Pharmacy",
        "Physics",
        "Psychology",
        "Public_Health",
        "Sociology",
    ]

    _DOMAIN_MAP = {
        "Accounting": "Business",
        "Agriculture": "Tech and Engineering",
        "Architecture_and_Engineering": "Tech and Engineering",
        "Art": "Art and Design",
        "Art_Theory": "Art and Design",
        "Basic_Medical_Science": "Health and Medicine",
        "Biology": "Science",
        "Chemistry": "Science",
        "Clinical_Medicine": "Health and Medicine",
        "Computer_Science": "Tech and Engineering",
        "Design": "Art and Design",
        "Diagnostics_and_Laboratory_Medicine": "Health and Medicine",
        "Economics": "Business",
        "Electronics": "Tech and Engineering",
        "Energy_and_Power": "Tech and Engineering",
        "Finance": "Business",
        "Geography": "Science",
        "History": "Humanities and Social Science",
        "Literature": "Humanities and Social Science",
        "Manage": "Business",
        "Marketing": "Business",
        "Materials": "Tech and Engineering",
        "Math": "Science",
        "Mechanical_Engineering": "Tech and Engineering",
        "Music": "Art and Design",
        "Pharmacy": "Health and Medicine",
        "Physics": "Science",
        "Psychology": "Humanities and Social Science",
        "Public_Health": "Health and Medicine",
        "Sociology": "Humanities and Social Science",
    }

    _SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer the question based on the image. "
        "For multiple-choice questions, respond with only the letter (A, B, C, etc.). "
        "For open questions, provide a concise answer."
    )

    def __init__(
        self,
        client: LLMClient,
        output_dir: str | Path,
        split: str = "dev",
        limit: int | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        force: bool = False,
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client (must support vision inputs).
            output_dir: Base output directory; results go to
                ``output_dir/mmmu/``.
            split: Dataset split to evaluate (``dev``, ``validation``,
                or ``test``). Default: ``dev``.
            limit: If set, cap the total number of evaluated samples
                across all subjects.
            max_tokens: Max new tokens for the answer generation.
            temperature: Sampling temperature.
            force: If ``True``, re-run even when cached JSONL exists.
        """
        super().__init__(client, output_dir, "mmmu", limit, force=force)
        self._split = split
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._limit: int = limit or 0

    def _build_messages(
        self,
        image_b64: str,
        question: str,
        question_type: str,
        options: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Build multimodal messages for an MMMU sample.

        Args:
            image_b64: Raw base64 string (without data URI prefix).
            question: Question text with ``<image 1>`` placeholders.
            question_type: ``"multiple-choice"`` or ``"short-answer"``.
            options: List of option strings for multiple-choice, or ``None``.

        Returns:
            OpenAI-compatible messages list with image content.
        """
        # Strip image placeholders from the question text
        clean_question = question.replace("<image 1>", "").strip()

        if question_type == "multiple-choice" and options:
            options_text = "\n".join(
                f"({chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
            )
            user_text = f"{clean_question}\n\n{options_text}\n\nAnswer:"
        else:
            user_text = f"{clean_question}\n\nAnswer:"

        data_uri = self._prepare_image_data_uri(image_b64)
        return self._build_vision_messages(data_uri, user_text, self._SYSTEM_PROMPT)

    @staticmethod
    def _extract_mc_answer(response: str, options: list[str]) -> str:
        """Extract a letter answer from model output for multiple-choice.

        Args:
            response: Raw model response.
            options: List of option strings.

        Returns:
            Uppercase letter (A, B, C...) or the raw cleaned text.
        """
        if not response:
            return ""

        text = BaseRunner._strip_thinking(response)
        num_options = len(options)
        all_choices = [chr(ord("A") + i) for i in range(num_options)]

        # Try explicit patterns first
        answer = BaseRunner._extract_letter_answer(
            text,
            patterns=[
                r"(?:answer|choice|option)[\s:：是为]*([A-Z])",
                r"\(?([A-Z])\)?",
                r"([A-Z])[.、)]",
            ],
            fallback=False,
            flags=re.IGNORECASE,
        )
        if answer and answer in all_choices:
            return answer

        # Fallback: look for any valid option letter
        cleaned = re.sub(r"[^A-Z]", "", text.strip(), flags=re.IGNORECASE)
        if len(cleaned) == 1 and cleaned.upper() in all_choices:
            return cleaned.upper()

        return text.strip()

    @staticmethod
    def _extract_open_answer(response: str) -> str:
        """Extract a concise answer from model output for open questions.

        Args:
            response: Raw model response.

        Returns:
            Cleaned answer string.
        """
        if not response:
            return ""

        text = BaseRunner._strip_thinking(response)
        text = text.strip()
        # Take the last line if multi-line, often the final answer
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[-1]
        return text

    def _compare_answer(self, pred: str, answer: str, question_type: str) -> bool:
        """Compare predicted and ground-truth answers.

        Args:
            pred: Predicted answer.
            answer: Ground-truth answer.
            question_type: ``"multiple-choice"`` or ``"short-answer"``.

        Returns:
            ``True`` if answers match.
        """
        if question_type == "multiple-choice":
            return pred.strip().upper() == answer.strip().upper()
        # For open questions, use normalized exact match
        return pred.strip().lower() == answer.strip().lower()

    def _predict_subject(
        self,
        subject: str,
        skip: int = 0,
        writer: _JsonlWriter | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference on a single MMMU subject.

        Args:
            subject: Subject name (e.g., ``"Accounting"``).
            skip: Number of samples to skip (already cached).
            writer: Optional streaming JSONL writer.

        Returns:
            List of prediction dicts.
        """

        dataset = load_dataset(
            "MMMU/MMMU",
            subject,
            split=self._split,
        )
        data = [dict(item) for item in dataset]

        logger.info(
            "Loaded MMMU/{} ({}) with {} rows",
            subject,
            self._split,
            len(data),
        )

        if skip:
            data = data[skip:]
            logger.info("Skipping {} cached samples for {}", skip, subject)

        results: list[dict[str, Any]] = []
        limit = min(self._limit, len(data)) or len(data)
        for i, item in enumerate(
            self._progress(data, desc=f"MMMU/{subject}", total=limit)
        ):
            # Skip multi-image samples (only evaluate single-image questions)
            has_multi_images = any(
                item.get(f"image_{i}") is not None for i in range(2, 8)
            )
            if has_multi_images:
                logger.debug("Skipping multi-image sample {}", item.get("id"))
                continue

            image = item.get("image_1")
            if image is None:
                logger.warning("Skipping sample {} with no image", item.get("id"))
                continue

            if limit <= i:
                break

            question_type = item.get("question_type", "multiple-choice")
            options = item.get("options")
            if isinstance(options, str):
                try:
                    options = ast.literal_eval(options)
                except Exception:
                    options = None

            image_valid = self._validate_image(image)
            if image_valid:
                messages = self._build_messages(
                    image,
                    item["question"],
                    question_type,
                    options,
                )
                response = self._chat(
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                if question_type == "multiple-choice":
                    pred = (
                        self._extract_mc_answer(response.content, options or [])
                        if response.valid
                        else ""
                    )
                else:
                    pred = (
                        self._extract_open_answer(response.content)
                        if response.valid
                        else ""
                    )
                finish_reason = response.finish_reason
                response_text = response.content
                valid = response.valid
            else:
                logger.warning("Skipping invalid image for {}", item.get("id"))
                pred = ""
                finish_reason = None
                response_text = ""
                valid = False

            answer = str(item.get("answer", "")).strip()
            record = {
                "id": item["id"],
                "subject": subject,
                "domain": self._DOMAIN_MAP.get(subject, "Unknown"),
                "question_type": question_type,
                "difficulty": item.get("topic_difficulty", ""),
                "subfield": item.get("subfield", ""),
                "pred": pred,
                "answer": answer,
                "correct": self._compare_answer(pred, answer, question_type)
                if valid
                else False,
                "valid": valid,
                "image_valid": image_valid,
                "finish_reason": finish_reason,
                "response": response_text,
            }
            results.append(record)
            if writer is not None:
                writer.write(record)

        return results

    def _compute_stats(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate accuracy by various dimensions.

        Args:
            data: Predictions from all subjects.

        Returns:
            Dictionary with ``overall``, ``by_domain``, ``by_subject``,
            ``by_question_type``, and ``by_difficulty`` keys.
        """
        # Overall stats (only valid samples)
        overall = self._overall_stats(data, correct_key="correct", valid_key="valid")

        # By domain
        by_domain = {}
        domains = set(self._DOMAIN_MAP.values())
        for domain in domains:
            domain_data = [
                item
                for item in data
                if item.get("domain") == domain and item.get("valid", True)
            ]
            if domain_data:
                by_domain[domain] = self._overall_stats(
                    domain_data, correct_key="correct", valid_key="valid"
                )

        # By subject
        by_subject = {}
        for subject in self._SUBJECTS:
            subject_data = [
                item
                for item in data
                if item.get("subject") == subject and item.get("valid", True)
            ]
            if subject_data:
                by_subject[subject] = self._overall_stats(
                    subject_data, correct_key="correct", valid_key="valid"
                )

        # By question type
        by_question_type = {}
        for qtype in ["multiple-choice", "short-answer"]:
            qtype_data = [
                item
                for item in data
                if item.get("question_type") == qtype and item.get("valid", True)
            ]
            if qtype_data:
                by_question_type[qtype] = self._overall_stats(
                    qtype_data, correct_key="correct", valid_key="valid"
                )

        # By difficulty
        by_difficulty = {}
        for diff in ["Easy", "Medium", "Hard"]:
            diff_data = [
                item
                for item in data
                if item.get("difficulty") == diff and item.get("valid", True)
            ]
            if diff_data:
                by_difficulty[diff] = self._overall_stats(
                    diff_data, correct_key="correct", valid_key="valid"
                )

        return {
            "overall": overall,
            "by_domain": by_domain,
            "by_subject": by_subject,
            "by_question_type": by_question_type,
            "by_difficulty": by_difficulty,
        }

    def dry_run(self, **kwargs: Any) -> None:
        """Load dataset and display sample images without API calls."""

        for subject in self._SUBJECTS:
            try:
                dataset = load_dataset(
                    "MMMU/MMMU",
                    subject,
                    split=self._split,
                )
                data = [dict(item) for item in dataset]

                logger.info(
                    "MMMU/{} ({}) — {} samples",
                    subject,
                    self._split,
                    len(data),
                )
                if not data:
                    continue

                self._inspect_dataset(
                    data,
                    label=f"MMMU/{subject}",
                    image_field="image_1",
                    fields=[
                        "id",
                        "question",
                        "question_type",
                        "options",
                        "answer",
                        "topic_difficulty",
                    ],
                )
            except Exception as exc:
                logger.warning(
                    "Failed to dry-run {}: {}",
                    subject,
                    exc,
                )
                continue

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the MMMU benchmark across all subjects.

        Returns:
            Dictionary with ``overall``, ``by_domain``, ``by_subject``,
            ``by_question_type``, and ``by_difficulty`` keys.
        """
        filename = "predictions.jsonl"
        existing, writer = self._resume_jsonl(filename)

        # Count existing records per subject for partial resume
        existing_by_subject: dict[str, int] = {}
        for rec in existing:
            s = rec.get("subject", "")
            existing_by_subject[s] = existing_by_subject.get(s, 0) + 1

        try:
            new_results: list[dict[str, Any]] = []
            for subject in self._SUBJECTS:
                skip = existing_by_subject.get(subject, 0)
                if skip:
                    logger.info("Skipping {} cached samples for {}", skip, subject)
                try:
                    subject_results = self._predict_subject(
                        subject, skip=skip, writer=writer
                    )
                    new_results.extend(subject_results)
                except Exception as exc:
                    logger.warning("Failed to evaluate {}: {}", subject, exc)
                    continue
        finally:
            writer.close()

        all_results = existing + new_results
        stats = self._compute_stats(all_results)
        o = stats["overall"]
        logger.info(
            "MMMU: {:.2f}% ({}/{})",
            o["accuracy"],
            o["correct"],
            o["total"],
        )
        for domain, s in stats["by_domain"].items():
            logger.info(
                "  {}: {:.2f}% ({}/{})",
                domain,
                s["accuracy"],
                s["correct"],
                s["total"],
            )
        return stats
