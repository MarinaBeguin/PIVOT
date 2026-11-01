"""Screening the requests of a public demo with Meta's Llama safeguard models.

Anyone can type into the public demo, and every request costs API calls. Before
the agent sees a request, an open-weight Meta model can screen it, on the
server's CPU and at no API cost:

- Llama Prompt Guard 2 (86M parameters, a classifier, milliseconds) flags
  attempts to hijack the agent ("ignore your instructions and...");
- Llama Guard 3 (1B parameters, a small language model, about a second)
  flags harmful requests, by the categories of its policy.

Set ``PIVOT_GUARD`` to ``prompt-guard``, ``llama-guard`` or both, comma
separated. The weights are gated on Hugging Face: request access to each
model, and give the server a token (``HF_TOKEN``). Unset, nothing is screened.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

PROMPT_GUARD = "meta-llama/Llama-Prompt-Guard-2-86M"
LLAMA_GUARD = "meta-llama/Llama-Guard-3-1B"
# Prompt Guard's labels for an attack (by version), and the probability above
# which a request is refused.
_ATTACK_LABELS = {"LABEL_1", "MALICIOUS", "JAILBREAK", "INJECTION"}
_ATTACK_THRESHOLD = 0.5

# Says why a request is refused, or None if it may go on.
Guard = Callable[[str], str | None]


def prompt_guard(classify: Callable[[str], list[dict[str, Any]]]) -> Guard:
    """Screens requests for attempts to hijack the agent.

    Args:
        classify: Text-classification pipeline of Llama Prompt Guard 2: text
            to ``[{"label": ..., "score": ...}]``.

    Returns:
        The guard.
    """
    def screen(request: str) -> str | None:
        """Screens one request.

        Args:
            request: The request in words.

        Returns:
            Why it is refused, or None.
        """
        best = classify(request)[0]
        attack = str(best["label"]).upper() in _ATTACK_LABELS
        if attack and float(best["score"]) > _ATTACK_THRESHOLD:
            return "This request looks like an attempt to change PIVOT's instructions."
        return None
    return screen


def llama_guard(generate: Callable[[str], str]) -> Guard:
    """Screens requests for harmful content.

    Args:
        generate: Llama Guard 3's verdict on a user message: ``"safe"``, or
            ``"unsafe"`` then the violated categories (``"S1,S9"``).

    Returns:
        The guard.
    """
    def screen(request: str) -> str | None:
        """Screens one request.

        Args:
            request: The request in words.

        Returns:
            Why it is refused, or None.
        """
        verdict = generate(request).strip().lower()
        if verdict.startswith("unsafe"):
            categories = verdict.removeprefix("unsafe").strip().upper() or "unspecified"
            return f"This request was flagged as unsafe (categories: {categories})."
        return None
    return screen


def combined(guards: list[Guard]) -> Guard:
    """Screens a request with several guards, the first refusal winning.

    Args:
        guards: The guards, cheapest first.

    Returns:
        The guard.
    """
    def screen(request: str) -> str | None:
        """Screens one request.

        Args:
            request: The request in words.

        Returns:
            Why it is refused, or None.
        """
        for guard in guards:
            if (reason := guard(request)) is not None:
                return reason
        return None
    return screen


def _prompt_guard_pipeline() -> Callable[[str], list[dict[str, Any]]]:
    """Loads Llama Prompt Guard 2 on the CPU.

    Returns:
        Its text-classification pipeline.
    """
    # Optional dependencies, installed where the guard is enabled.
    # pylint: disable-next=import-outside-toplevel,import-error
    from transformers import pipeline  # type: ignore[import-not-found]

    classify: Callable[[str], list[dict[str, Any]]] = pipeline(
        "text-classification", model=PROMPT_GUARD, device="cpu", truncation=True)
    return classify


def _llama_guard_generate() -> Callable[[str], str]:
    """Loads Llama Guard 3 1B on the CPU.

    Returns:
        A function from a user message to the model's verdict.
    """
    # Optional dependencies, installed where the guard is enabled.
    # pylint: disable=import-outside-toplevel,import-error
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # pylint: enable=import-outside-toplevel,import-error
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_GUARD)
    model = AutoModelForCausalLM.from_pretrained(LLAMA_GUARD, torch_dtype=torch.bfloat16).eval()

    def generate(request: str) -> str:
        """Asks Llama Guard for its verdict on a user message.

        Args:
            request: The user message.

        Returns:
            The verdict, e.g. ``"safe"`` or ``"unsafe\nS1"``.
        """
        conversation = [{"role": "user", "content": [{"type": "text", "text": request}]}]
        inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(inputs, max_new_tokens=12, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        return str(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
    return generate


def guard_from_env() -> Guard | None:
    """Builds the guard named by ``PIVOT_GUARD``.

    Returns:
        The guard, or None if no request is screened.

    Raises:
        ValueError: If ``PIVOT_GUARD`` names an unknown guard.
    """
    names = [n.strip() for n in os.environ.get("PIVOT_GUARD", "").split(",") if n.strip()]
    loaders: dict[str, Callable[[], Guard]] = {
        "prompt-guard": lambda: prompt_guard(_prompt_guard_pipeline()),
        "llama-guard": lambda: llama_guard(_llama_guard_generate()),
    }
    unknown = set(names) - loaders.keys()
    if unknown:
        raise ValueError(f"Unknown PIVOT_GUARD {sorted(unknown)}; use {sorted(loaders)}.")
    # The cheap classifier first: most refusals need no language model.
    ordered = sorted(names, key=list(loaders).index)
    return combined([loaders[name]() for name in ordered]) if ordered else None
