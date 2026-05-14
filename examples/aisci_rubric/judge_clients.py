"""Pluggable LLM judge clients for the aisci_rubric task.

Three backends, all OpenAI-compatible /v1/chat/completions:
  - hf:     HuggingFace dedicated inference endpoint
  - nvidia: https://inference-api.nvidia.com/v1
  - local:  self-hosted SGLang server (URL handed off via $JUDGE_BASE_URL)

All concurrency, retry, and timeout policy lives here so reward_fn.py stays
focused on rubric scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import yaml


@dataclass
class JudgeConfig:
    backend: str
    base_url: str | None
    model: str
    api_key_env: str | None
    sampling: dict[str, Any]
    concurrency: int
    max_attempts: int
    backoff_seconds: list[int]
    timeout_seconds: int

    @classmethod
    def from_dict(cls, raw: dict) -> "JudgeConfig":
        retries = raw.get("retries", {}) or {}
        return cls(
            backend=raw["backend"],
            base_url=raw.get("base_url"),
            model=raw["model"],
            api_key_env=raw.get("api_key_env"),
            sampling=raw.get("sampling") or {},
            concurrency=int(raw.get("concurrency", 32)),
            max_attempts=int(retries.get("max_attempts", 3)),
            backoff_seconds=list(retries.get("backoff_seconds", [15, 30, 60])),
            timeout_seconds=int(retries.get("timeout_seconds", 900)),
        )


def load_judge_config(path: str | os.PathLike) -> JudgeConfig:
    with open(path) as f:
        return JudgeConfig.from_dict(yaml.safe_load(f))


def _resolve_base_url(cfg: JudgeConfig) -> str:
    if cfg.base_url:
        return cfg.base_url.rstrip("/")
    # Local backend: URL handed off by serve_judge_local.sh via env.
    fallback = os.environ.get("JUDGE_BASE_URL")
    if not fallback:
        raise RuntimeError(
            f"backend={cfg.backend!r} has base_url=null and JUDGE_BASE_URL is unset; "
            "serve_judge_local.sh must export JUDGE_BASE_URL before training starts."
        )
    return fallback.rstrip("/")


def _resolve_api_key(cfg: JudgeConfig) -> str | None:
    if not cfg.api_key_env:
        return None
    key = os.environ.get(cfg.api_key_env)
    if not key:
        raise RuntimeError(f"backend={cfg.backend!r} requires env var {cfg.api_key_env}, which is unset")
    return key


class JudgeClient:
    """Async OpenAI-compatible chat-completions client with bounded concurrency + retries."""

    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self._session: aiohttp.ClientSession | None = None
        self._base_url = _resolve_base_url(cfg)
        self._api_key = _resolve_api_key(cfg)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def chat(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Single chat-completion call. Returns (text, runtime_metrics)."""
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        sampling = dict(self.cfg.sampling)
        # OpenAI new responses API uses max_output_tokens; chat-completions uses max_tokens.
        if "max_output_tokens" in sampling and "max_tokens" not in sampling:
            sampling["max_tokens"] = sampling.pop("max_output_tokens")

        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            **sampling,
        }

        async with self._sem:
            return await self._send_with_retries(url, headers, payload)

    async def _send_with_retries(self, url, headers, payload) -> tuple[str, dict[str, Any]]:
        session = await self._ensure_session()
        last_err: Exception | None = None
        for attempt in range(1, self.cfg.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    body = await resp.text()
                    if resp.status >= 500 or resp.status == 429:
                        raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")
                    if resp.status >= 400:
                        # 4xx other than 429 is unlikely to recover — surface immediately.
                        raise RuntimeError(f"HTTP {resp.status} (non-retryable): {body[:500]}")
                    data = json.loads(body)
                    # Coerce None → "" at the source. Some OpenAI-compatible
                    # backends (incl. NVIDIA's gpt-oss-20b) return content=null
                    # when the model emits a refusal / tool_call / empty body.
                    # Downstream parsers expect a string.
                    text = data["choices"][0]["message"].get("content") or ""
                    usage = data.get("usage") or {}
                    metrics = {
                        "judge/latency_seconds": time.perf_counter() - t0,
                        "judge/prompt_tokens": usage.get("prompt_tokens", 0),
                        "judge/completion_tokens": usage.get("completion_tokens", 0),
                        "judge/attempts": attempt,
                    }
                    return text, metrics
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                last_err = e
                if attempt >= self.cfg.max_attempts:
                    break
                wait = self.cfg.backoff_seconds[min(attempt - 1, len(self.cfg.backoff_seconds) - 1)]
                print(
                    f"[judge_clients] attempt {attempt}/{self.cfg.max_attempts} failed ({type(e).__name__}: {e}); "
                    f"backing off {wait}s",
                    file=sys.stderr,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(f"judge call failed after {self.cfg.max_attempts} attempts: {last_err}")


def make_judge(cfg: JudgeConfig) -> JudgeClient:
    return JudgeClient(cfg)


# ── Smoke entry point ─────────────────────────────────────────────────────────
# python -m examples.aisci_rubric.judge_clients --config <path>
# Sends a single fixed (query, response, rubric) triple through the configured
# backend and prints the parsed n_yes/n_criteria, so we can validate creds and
# routing without spinning up the trainer.

_SMOKE_PROMPT = """You are an expert grader. Output exactly one ---SCORE--- block:

---SCORE---
{"criterion_id": 1, "judgment": "yes", "rationale": "smoke ping"}
"""


async def _smoke(config_path: str):
    cfg = load_judge_config(config_path)
    print(f"backend={cfg.backend} base_url={cfg.base_url} model={cfg.model}")
    client = make_judge(cfg)
    try:
        text, metrics = await client.chat(_SMOKE_PROMPT)
        print("=== response ===")
        print(text)
        print("=== metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to judge_*.yaml")
    args = parser.parse_args()
    asyncio.run(_smoke(args.config))
