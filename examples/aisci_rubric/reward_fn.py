"""Custom reward function for the aisci_rubric task.

Plugged into miles via:
  --custom-rm-path examples.aisci_rubric.reward_fn.reward_fn
  --reward-key reward_value

Each Sample carries the raw query and parsed rubric in metadata (set by
HFRolloutDataSource). The reward function formats a judge prompt, hits the
configured backend, parses ---SCORE--- delimited JSON, and returns a dict:
    {reward_value, reward_cat, n_yes, n_criteria, judge_latency_seconds, ...}
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from miles.utils.types import Sample

# Resolved against the miles repo root (CWD when the trainer runs).
_REPO_ROOT = Path(__file__).resolve().parents[2]

_PLACEHOLDER_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")
_SCORE_DELIMITER = "---SCORE---"
_MAX_RESPONSE_CHARS = 12000
_THINK_END = "</think>"
_THINK_START = "<think>"

# Lazily initialized singletons. RM is invoked many times per rollout — we
# don't want to reload the prompt template or open a new aiohttp session per
# sample. The first call wins; subsequent calls reuse.
_judge_client = None
_judge_cfg = None
_prompt_template: str | None = None
_length_discount_factor: float = 1.0
_init_lock = asyncio.Lock()


async def _ensure_init():
    global _judge_client, _judge_cfg, _prompt_template, _length_discount_factor
    if _judge_client is not None:
        return
    async with _init_lock:
        if _judge_client is not None:
            return
        # Deferred import: judge_clients pulls in aiohttp / yaml; we want
        # ImportError to surface here, not at module-import time inside Ray.
        from examples.aisci_rubric.judge_clients import load_judge_config, make_judge

        cfg_path = os.environ.get("AISCI_JUDGE_CONFIG")
        if not cfg_path:
            raise RuntimeError(
                "AISCI_JUDGE_CONFIG is unset. The trainer launcher must export this "
                "before submitting the ray job."
            )
        _judge_cfg = load_judge_config(cfg_path)
        _judge_client = make_judge(_judge_cfg)

        import yaml
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        prompt_rel = raw.get("prompt_path")
        if not prompt_rel:
            raise RuntimeError(f"judge config {cfg_path} missing 'prompt_path'")
        p = Path(prompt_rel)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f"judge prompt template not found: {p}")
        _prompt_template = p.read_text(encoding="utf-8")

        ld = raw.get("length_discount") or {}
        _length_discount_factor = float(ld.get("factor", 1.0))


def _truncate(text: str, max_chars: int = _MAX_RESPONSE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]"


def _format_rubric(criteria: list[dict]) -> str:
    lines = []
    for c in criteria:
        cid = c.get("criterion_id", "")
        aspect = c.get("aspect", "")
        criterion = c.get("criterion", "")
        lines.append(f"{cid}. [{aspect}] {criterion}")
    return "\n".join(lines)


def _parse_rubric(rubric_raw: Any) -> list[dict]:
    if isinstance(rubric_raw, list):
        return rubric_raw
    if isinstance(rubric_raw, str):
        try:
            parsed = json.loads(rubric_raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def _fill_template(template: str, **kwargs) -> str:
    def _replace(m):
        return str(kwargs.get(m.group(1), m.group(0)))
    return _PLACEHOLDER_RE.sub(_replace, template)


def _parse_score_blocks(raw_text: str | None) -> list[dict]:
    """Parse ---SCORE--- delimited JSON judgment objects."""
    if not raw_text:
        return []
    out = []
    for part in raw_text.split(_SCORE_DELIMITER):
        part = part.strip()
        if not part:
            continue
        json_str = re.sub(r"```(?:json)?\s*", "", part).strip()
        try:
            s = json.loads(json_str)
            if isinstance(s, dict) and "judgment" in s:
                out.append(s)
                continue
        except json.JSONDecodeError:
            pass
        # Fallback: scan brace-matched JSON dicts inside the block.
        depth = 0
        start = None
        for i, ch in enumerate(json_str):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        s = json.loads(json_str[start : i + 1])
                        if isinstance(s, dict) and "judgment" in s:
                            out.append(s)
                    except json.JSONDecodeError:
                        pass
                    start = None
    return out


def _compute_score(scores: list[dict], n_criteria: int) -> tuple[float, int]:
    n_yes = 0
    for s in scores:
        j = str(s.get("judgment", "")).lower().strip()
        if j in ("yes", "y", "true", "1"):
            n_yes += 1
    if n_criteria <= 0:
        return 0.0, 0
    return max(0.0, min(1.0, n_yes / n_criteria)), n_yes


def _zero_reward(reward_cat: str, n_criteria: int) -> dict:
    return {
        "reward_value": 0.0,
        "reward_cat": reward_cat,
        "n_yes": 0,
        "n_criteria": n_criteria,
        "judge_latency_seconds": 0.0,
    }


def _extract_visible(response: str) -> str:
    """Return the substring after the last </think>, or the full response if
    there's no </think> at all. Caller must separately guard against the
    "<think> opened but never closed" case (no visible content emitted)."""
    if _THINK_END in response:
        return response.rsplit(_THINK_END, 1)[1]
    return response


async def _score_one(sample: Sample) -> dict:
    """Score a single sample. Returns the reward dict that's stored on Sample.reward."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    query = metadata.get("query") or ""
    rubric_raw = metadata.get("rubric_criteria") or metadata.get("rubric") or []
    rubric = _parse_rubric(rubric_raw)
    response = sample.response or ""

    if not query.strip() or not rubric:
        return _zero_reward("no_input", len(rubric))

    # Truncated rollouts get 0 regardless of WHERE the cut landed (mid-thinking
    # or mid-final-response). A truncated output is incomplete by definition,
    # and without this guard the model has no incentive to keep <think> short
    # enough to fit visible content inside the rollout budget.
    if sample.status == Sample.Status.TRUNCATED:
        return _zero_reward("truncated", len(rubric))

    # If the model emitted <think> but never closed it (no </think>), there is
    # no visible content to score — this shouldn't happen when status != TRUNCATED
    # (the model would only stop mid-thinking via an EOS), but guard anyway.
    if _THINK_START in response and _THINK_END not in response:
        return _zero_reward("unclosed_thinking", len(rubric))

    candidate = _extract_visible(response).strip()
    if not candidate:
        return _zero_reward("no_visible_content", len(rubric))

    prompt = _fill_template(
        _prompt_template,
        QUERY=query,
        CANDIDATE_RESPONSE=_truncate(candidate),
        RUBRIC=_format_rubric(rubric),
    )

    try:
        judge_text, metrics = await _judge_client.chat(prompt)
    except Exception as e:
        return {
            "reward_value": 0.0,
            "reward_cat": "judge_error",
            "n_yes": 0,
            "n_criteria": len(rubric),
            "judge_latency_seconds": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }

    parsed = _parse_score_blocks(judge_text)
    if not parsed:
        return {
            "reward_value": 0.0,
            "reward_cat": "no_score",
            "n_yes": 0,
            "n_criteria": len(rubric),
            "judge_latency_seconds": metrics.get("judge/latency_seconds", 0.0),
        }

    score, n_yes = _compute_score(parsed, len(rubric))

    response_len_tokens = sample.response_length or len(response.split())
    if _length_discount_factor != 1.0:
        score *= _length_discount_factor**response_len_tokens

    return {
        "reward_value": float(score),
        "reward_cat": "ok",
        "n_yes": int(n_yes),
        "n_criteria": len(rubric),
        "judge_latency_seconds": float(metrics.get("judge/latency_seconds", 0.0)),
        "judge_prompt_tokens": int(metrics.get("judge/prompt_tokens", 0)),
        "judge_completion_tokens": int(metrics.get("judge/completion_tokens", 0)),
    }


async def reward_fn(args, sample_or_samples, **kwargs):
    """Custom RM entry point.

    miles dispatches differently depending on the call site:
      - batched_async_rm passes a list[Sample]
      - async_rm passes a single Sample
    Detect input shape and handle both.
    """
    await _ensure_init()
    if isinstance(sample_or_samples, list):
        return await asyncio.gather(*[_score_one(s) for s in sample_or_samples])
    return await _score_one(sample_or_samples)


# ── Smoke entry point ─────────────────────────────────────────────────────────
# AISCI_JUDGE_CONFIG=examples/aisci_rubric/configs/judge_hf.yaml \
#   python -m examples.aisci_rubric.reward_fn

if __name__ == "__main__":
    from types import SimpleNamespace

    fake = Sample()
    fake.prompt = "What is the boiling point of water at sea level in Celsius?"
    fake.response = "Water boils at 100 degrees Celsius at sea level (101.325 kPa)."
    fake.response_length = 14
    fake.metadata = {
        "query": fake.prompt,
        "rubric_criteria": [
            {"criterion_id": 1, "aspect": "correctness", "criterion": "States 100°C as the boiling point."},
            {"criterion_id": 2, "aspect": "completeness", "criterion": "Mentions standard atmospheric pressure."},
        ],
    }

    async def _go():
        out = await reward_fn(SimpleNamespace(), [fake])
        print(out)

    asyncio.run(_go())
