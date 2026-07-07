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


async def _judge_against_rubric(query: str, candidate: str, rubric: list[dict]) -> dict:
    """Judge one candidate answer against ONE rubric (the single-rubric core,
    shared by the single-answer and multi-answer paths). Returns:
        {score, n_yes, n_criteria, latency, cat}
    where cat in {ok, no_input, judge_error, no_score}. Never raises."""
    n_criteria = len(rubric)
    if n_criteria <= 0:
        return {"score": 0.0, "n_yes": 0, "n_criteria": 0, "latency": 0.0, "cat": "no_input"}

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
            "score": 0.0, "n_yes": 0, "n_criteria": n_criteria, "latency": 0.0,
            "cat": "judge_error", "error": f"{type(e).__name__}: {e}",
        }

    latency = float(metrics.get("judge/latency_seconds", 0.0))
    parsed = _parse_score_blocks(judge_text)
    if not parsed:
        return {"score": 0.0, "n_yes": 0, "n_criteria": n_criteria, "latency": latency, "cat": "no_score"}

    score, n_yes = _compute_score(parsed, n_criteria)
    return {
        "score": float(score), "n_yes": int(n_yes), "n_criteria": n_criteria,
        "latency": latency, "cat": "ok",
        "judge_prompt_tokens": int(metrics.get("judge/prompt_tokens", 0)),
        "judge_completion_tokens": int(metrics.get("judge/completion_tokens", 0)),
    }


def _collect_answers(metadata: dict) -> list[dict] | None:
    """Multi-answer (M2) mode: metadata carries an `answers` list, each answer
    with its OWN rubric. Returns the list of {answer_id, source, rubric} answers
    that have a non-empty rubric, or None when not in multi-answer mode."""
    answers_raw = metadata.get("answers")
    if not isinstance(answers_raw, list) or not answers_raw:
        return None
    out = []
    for a in answers_raw:
        if not isinstance(a, dict):
            continue
        rubric = _parse_rubric(a.get("rubric"))
        if rubric:
            out.append({"answer_id": a.get("answer_id"), "source": a.get("source"), "rubric": rubric})
    return out


async def _score_one(sample: Sample) -> dict:
    """Score a single sample. Returns the reward dict that's stored on Sample.reward."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    query = metadata.get("query") or ""
    response = sample.response or ""

    # Multi-answer mode (M2): reward = max over each answer's rubric (no value
    # weight). Single-answer mode (unchanged): one rubric in rubric_criteria/rubric.
    answers = _collect_answers(metadata)
    multi = answers is not None
    if multi:
        rubrics = [a["rubric"] for a in answers]
        n_criteria_repr = max((len(r) for r in rubrics), default=0)
        has_rubric = bool(rubrics)
    else:
        rubric = _parse_rubric(metadata.get("rubric_criteria") or metadata.get("rubric") or [])
        n_criteria_repr = len(rubric)
        has_rubric = bool(rubric)

    # ── Response-level guards (computed ONCE; identical to the single-answer path) ──
    if not query.strip() or not has_rubric:
        return _zero_reward("no_input", n_criteria_repr)

    # Truncated rollouts get 0 regardless of WHERE the cut landed (mid-thinking
    # or mid-final-response). A truncated output is incomplete by definition,
    # and without this guard the model has no incentive to keep <think> short
    # enough to fit visible content inside the rollout budget.
    if sample.status == Sample.Status.TRUNCATED:
        return _zero_reward("truncated", n_criteria_repr)

    # Strict thinking-completion gate (opt-in via the data config's
    # `require_think_close: true`, surfaced into metadata by hf_data_source).
    # When the assistant prompt ends with "<think>\n" (enable_thinking=true),
    # the response has NO opening <think> tag — so the `_THINK_START in response`
    # guard below never fires for a thinking rollout. A COMPLETED response that
    # ran out of thoughts and emitted EOS without ever writing </think> has no
    # finished answer; _extract_visible would otherwise treat the raw thinking
    # as the answer. Require a closed </think> so reward only flows to responses
    # that finished within budget AND produced an answer after </think>.
    if bool(metadata.get("require_think_close", False)) and _THINK_END not in response:
        return _zero_reward("no_think_close", n_criteria_repr)

    # If the model emitted <think> but never closed it (no </think>), there is
    # no visible content to score — this shouldn't happen when status != TRUNCATED
    # (the model would only stop mid-thinking via an EOS), but guard anyway.
    if _THINK_START in response and _THINK_END not in response:
        return _zero_reward("unclosed_thinking", n_criteria_repr)

    candidate = _extract_visible(response).strip()
    if not candidate:
        return _zero_reward("no_visible_content", n_criteria_repr)

    response_len_tokens = sample.response_length or len(response.split())
    discount = _length_discount_factor**response_len_tokens if _length_discount_factor != 1.0 else 1.0

    # ── Single-answer path — behavior byte-for-byte unchanged ─────────────────
    if not multi:
        r = await _judge_against_rubric(query, candidate, rubric)
        if r["cat"] == "judge_error":
            return {
                "reward_value": 0.0, "reward_cat": "judge_error", "n_yes": 0,
                "n_criteria": r["n_criteria"], "judge_latency_seconds": 0.0, "error": r.get("error", ""),
            }
        if r["cat"] == "no_score":
            return {
                "reward_value": 0.0, "reward_cat": "no_score", "n_yes": 0,
                "n_criteria": r["n_criteria"], "judge_latency_seconds": r["latency"],
            }
        return {
            "reward_value": float(r["score"] * discount),
            "reward_cat": "ok",
            "n_yes": int(r["n_yes"]),
            "n_criteria": int(r["n_criteria"]),
            "judge_latency_seconds": float(r["latency"]),
            "judge_prompt_tokens": int(r.get("judge_prompt_tokens", 0)),
            "judge_completion_tokens": int(r.get("judge_completion_tokens", 0)),
        }

    # ── Multi-answer path (M2) — judge each answer's rubric, reduce with MAX ───
    results = await asyncio.gather(*[_judge_against_rubric(query, candidate, a["rubric"]) for a in answers])
    # Prefer a successfully-judged answer (cat ok), then the higher score; ties
    # at score 0 fall back to whatever judged so reward_cat surfaces the failure.
    best_i = max(range(len(results)), key=lambda i: (results[i]["cat"] == "ok", results[i]["score"]))
    best = results[best_i]
    best_ans = answers[best_i]
    seed_score = next((results[i]["score"] for i, a in enumerate(answers) if a.get("source") == "seed"), None)
    return {
        "reward_value": float(best["score"] * discount),
        "reward_cat": best["cat"],            # "ok" when the winning answer judged cleanly
        "n_yes": int(best["n_yes"]),
        "n_criteria": int(best["n_criteria"]),
        "judge_latency_seconds": float(sum(r["latency"] for r in results)),
        "n_answers": len(answers),
        "best_answer_id": best_ans.get("answer_id"),
        "best_source": best_ans.get("source"),
        "max_score": float(best["score"]),
        "seed_score": (float(seed_score) if seed_score is not None else None),
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

    # Multi-answer (M2) fake: same response, two answers with disjoint rubrics.
    # The seed rubric (boiling point) is satisfied -> score 1.0; the "similar"
    # rubric (altitude/pressure-cooker) is not -> score 0.0. max -> 1.0, won by
    # the seed (best_source="seed", best_answer_id=0).
    fake_multi = Sample()
    fake_multi.prompt = fake.prompt
    fake_multi.response = fake.response
    fake_multi.response_length = 14
    fake_multi.metadata = {
        "query": fake.prompt,
        "answers": [
            {"answer_id": 0, "source": "seed", "rubric": [
                {"criterion_id": 1, "aspect": "correctness", "criterion": "States 100°C as the boiling point."},
            ]},
            {"answer_id": 1, "source": "similar", "rubric": [
                {"criterion_id": 1, "aspect": "alt", "criterion": "Explains how boiling point drops at high altitude."},
                {"criterion_id": 2, "aspect": "alt", "criterion": "Mentions using a pressure cooker."},
            ]},
        ],
    }

    async def _go():
        single = await reward_fn(SimpleNamespace(), [fake])
        print("single-answer:", single)
        multi = await reward_fn(SimpleNamespace(), [fake_multi])
        print("multi-answer :", multi)
        m = multi[0]
        assert m.get("reward_cat") == "ok", m
        assert m.get("best_source") == "seed" and m.get("best_answer_id") == 0, m
        assert abs(m.get("reward_value", 0.0) - m.get("max_score", -1)) < 1e-9, m
        print("multi-answer smoke OK: reward = max, won by seed")

    asyncio.run(_go())
