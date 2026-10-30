"""Meta Model API client: configuration and resilient calls.

The API is OpenAI-compatible, so the official ``openai`` SDK is used with
Meta's base URL. ``PIVOT_BACKEND`` chooses between two versions of PIVOT:
``api`` (Muse and SAM 3.1 through the Meta API, with a key) and ``open``
(open weights on one's own GPU: Llama served by vLLM, SAM 3.1 self-hosted).

As of Oct 27, 2026 the API intermittently answers ``model_not_found`` for
models the account does have (about half of the calls in our measurements);
such errors are retried with exponential backoff.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import NotFoundError, OpenAI

T = TypeVar("T")

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
# Defaults of the open-weight version (see scripts/cluster/llama_server.sbatch).
OPEN_BASE_URL = "http://localhost:8000/v1"
OPEN_AGENT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass(frozen=True)
class MetaConfig:
    """Connection settings and model choices.

    Attributes:
        api_key: Meta Model API key.
        base_url: API base URL.
        agent_model: Model driving the agent (tool calling).
        vision_model: Model naming the objects seen in keyframes.
        sam_model: Segmentation model.
        reasoning_effort: Reasoning effort requested from the agent model;
            None for models that do not take it (Llama served by vLLM).
    """

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    agent_model: str = "muse-spark-1.3"
    vision_model: str = "muse-spark-1.3"
    sam_model: str = "sam-3.1"
    reasoning_effort: str | None = "low"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> MetaConfig:
        """Reads the configuration from the environment.

        Variables: ``PIVOT_BACKEND`` (``api`` by default, or ``open``),
        ``META_API_KEY`` (required by ``api``), ``META_API_BASE_URL``,
        ``PIVOT_AGENT_MODEL``, ``PIVOT_VISION_MODEL``, ``PIVOT_SAM_MODEL``,
        ``PIVOT_REASONING_EFFORT`` (empty for none). With ``open``, the agent
        defaults to Llama on a local vLLM server (``PIVOT_OPEN_URL``); any
        variable set explicitly still wins.

        Args:
            env_file: Optional ``.env`` file loaded first; existing variables win.

        Returns:
            The configuration.

        Raises:
            RuntimeError: If no API key is set.
        """
        load_dotenv(env_file)
        if os.environ.get("PIVOT_BACKEND", "api").lower() == "open":
            # A local server needs no key; Llama takes no reasoning effort.
            api_key = os.environ.get("META_API_KEY") or "local"
            base_url = os.environ.get("PIVOT_OPEN_URL", OPEN_BASE_URL)
            agent_model, effort = OPEN_AGENT_MODEL, ""
        else:
            api_key = os.environ.get("META_API_KEY", "")
            if not api_key:
                raise RuntimeError("META_API_KEY is not set; copy .env.example to .env, "
                                   "or set PIVOT_BACKEND=open for the open-weight version.")
            base_url = os.environ.get("META_API_BASE_URL", DEFAULT_BASE_URL)
            agent_model, effort = cls.agent_model, "low"
        return cls(
            api_key=api_key,
            base_url=base_url,
            agent_model=os.environ.get("PIVOT_AGENT_MODEL", agent_model),
            vision_model=os.environ.get("PIVOT_VISION_MODEL", cls.vision_model),
            sam_model=os.environ.get("PIVOT_SAM_MODEL", cls.sam_model),
            reasoning_effort=os.environ.get("PIVOT_REASONING_EFFORT", effort) or None,
        )

    def client(self, timeout: float = 300.0) -> OpenAI:
        """Creates an SDK client for the Meta Model API.

        Args:
            timeout: Request timeout, in seconds.

        Returns:
            The client. Rate limits and server errors are retried by the SDK.
        """
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout,
                      max_retries=4)


@dataclass
class Usage:
    """Model calls and tokens spent, accumulated over requests.

    Attributes:
        calls: Number of completed requests.
        prompt_tokens: Input tokens.
        completion_tokens: Output tokens, reasoning included.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, response: Any) -> None:
        """Counts one response.

        Args:
            response: A Chat Completions response; its ``usage`` may be missing.
        """
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0


def is_spurious_not_found(error: Exception) -> bool:
    """Tells whether an error is the API's intermittent ``model_not_found``.

    Args:
        error: Exception raised by an SDK call.

    Returns:
        True for retryable ``model_not_found`` errors.
    """
    if not isinstance(error, NotFoundError):
        return False
    return error.code == "model_not_found" or "model was not found" in str(error)


def call_with_retry(call: Callable[[], T], attempts: int = 6, base_delay: float = 0.5,
                    max_delay: float = 8.0, sleep: Callable[[float], None] = time.sleep) -> T:
    """Runs an API call, retrying spurious ``model_not_found`` errors.

    Args:
        call: Zero-argument function performing the request.
        attempts: Maximum number of attempts.
        base_delay: Delay before the first retry, doubled at each retry, in seconds.
        max_delay: Upper bound of the delay between retries, in seconds.
        sleep: Sleep function, injectable for tests.

    Returns:
        The call's result.

    Raises:
        NotFoundError: If every attempt failed with ``model_not_found``.
    """
    for attempt in range(attempts - 1):
        try:
            return call()
        except NotFoundError as error:
            if not is_spurious_not_found(error):
                raise
            sleep(min(max_delay, base_delay * 2**attempt))
    return call()
