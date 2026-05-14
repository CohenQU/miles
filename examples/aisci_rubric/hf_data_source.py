"""HF-streaming data source for the aisci_rubric task (and reusable for other rubric tasks).

Pluggable into miles via:
  --data-source-path examples.aisci_rubric.hf_data_source.HFRolloutDataSource

Reads a YAML config (path = AISCI_DATA_CONFIG env var) describing:
  - HF Hub repo_id and split names
  - column → Sample-field mapping (query → prompt, rubric → metadata, ...)

Subclasses RolloutDataSourceWithBuffer so it inherits offset checkpointing,
buffer / get_samples logic; only __init__ is overridden to populate
self.dataset from HF Hub instead of from a JSONL.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import yaml

from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class _HFDataset:
    """Thin shim that exposes the (samples, origin_samples, shuffle, __len__)
    interface that miles.rollout.data_source.RolloutDataSource.get_samples
    expects, without needing the JSONL-backed miles.utils.data.Dataset."""

    def __init__(self, samples: list[Sample], seed: int = 42):
        self.origin_samples = samples
        self.samples = list(samples)
        self.seed = seed
        self.epoch_id = -1

    def shuffle(self, new_epoch_id: int):
        if self.epoch_id == new_epoch_id:
            return
        rng = random.Random(self.seed + new_epoch_id)
        idxs = list(range(len(self.origin_samples)))
        rng.shuffle(idxs)
        self.samples = [self.origin_samples[i] for i in idxs]
        self.epoch_id = new_epoch_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def _parse_rubric(raw: Any) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            return []
    return []


class HFRolloutDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        # Skip parent's JSONL-loading branch but keep its bookkeeping fields.
        # We reach into the grandparent (RolloutDataSource) only to set up the
        # state vars; we do NOT call its __init__, which would try to load
        # args.prompt_data via the file-backed Dataset.
        self.args = args
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata = {}
        # WithBuffer extras:
        self.buffer = []
        from miles.rollout.data_source import pop_first
        from miles.utils.misc import load_function
        self.buffer_filter = (
            pop_first if args.buffer_filter_path is None else load_function(args.buffer_filter_path)
        )

        cfg_path = os.environ.get("AISCI_DATA_CONFIG")
        if not cfg_path:
            raise RuntimeError(
                "AISCI_DATA_CONFIG is unset. The trainer launcher must export this "
                "before submitting the ray job."
            )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        repo_id: str = cfg["repo_id"]
        split: str = cfg.get("train_split", "train")
        trust_remote_code: bool = bool(cfg.get("trust_remote_code", True))
        field_map: dict[str, str] = cfg.get("field_map") or {}
        filters: dict[str, bool] = cfg.get("filters") or {}
        apply_chat_template: bool = bool(cfg.get("apply_chat_template", True))

        query_col = field_map.get("query", "query")
        rubric_col = field_map.get("rubric", "rubric_criteria")
        title_col = field_map.get("title", "title")

        # Tokenizer for chat template (mirrors what miles.utils.data.Dataset does).
        tokenizer = load_tokenizer(
            args.hf_checkpoint,
            chat_template_path=args.chat_template_path,
            trust_remote_code=True,
        )

        # Deferred import: `datasets` is a heavy dep, only pull it in when this
        # source is actually used.
        from datasets import load_dataset

        logger.info(f"[aisci_rubric] loading {repo_id} split={split}")
        ds = load_dataset(repo_id, split=split, trust_remote_code=trust_remote_code)

        samples: list[Sample] = []
        skipped = {"empty_query": 0, "empty_rubric": 0}
        for row in ds:
            query = row.get(query_col, "") or ""
            rubric_raw = row.get(rubric_col, "")
            title = row.get(title_col, "")

            if filters.get("drop_empty_query", True) and not query.strip():
                skipped["empty_query"] += 1
                continue
            rubric = _parse_rubric(rubric_raw)
            if filters.get("drop_empty_rubric", True) and not rubric:
                skipped["empty_rubric"] += 1
                continue

            if apply_chat_template:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": query}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = query

            samples.append(
                Sample(
                    prompt=prompt,
                    label=None,
                    metadata={
                        "query": query,
                        "rubric_criteria": rubric,
                        "title": title,
                        "rm_type": "aisci_rubric",  # advisory, custom-rm-path takes precedence
                    },
                )
            )

        logger.info(
            f"[aisci_rubric] loaded {len(samples)} samples from {repo_id}/{split} "
            f"(skipped: {skipped})"
        )
        if not samples:
            raise RuntimeError(
                f"no samples loaded from {repo_id}/{split} — check field_map and filters"
            )

        self.dataset = _HFDataset(samples, seed=args.rollout_seed)
        if getattr(args, "rollout_shuffle", False):
            self.dataset.shuffle(self.epoch_id)
