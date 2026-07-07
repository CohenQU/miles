#!/usr/bin/env python3
"""Add a base model's non-LM weights (e.g. model.visual.*, mtp.*) into a converted
LM-only HF checkpoint, producing a COMPLETE model that vLLM/transformers can load.

The miles torch_dist→HF convert exports only the language model (model.language_model.*
+ lm_head) — for a VL base (Qwen3.6-27B) the vision tower (model.visual.*) and MTP
heads are dropped, so the VL architecture won't load. Those weights are frozen during
RL (unchanged from base), so we copy them straight from the base HF dir.

Usage: python3 add_base_nonlm_weights.py <converted_hf_dir> <base_hf_dir>
Idempotent: if all base keys are already present, or the extra shard exists, it no-ops.
Requires torch + safetensors (run in the miles container).
"""
import json, os, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file

conv_dir, base_dir = sys.argv[1], sys.argv[2]
ci = os.path.join(conv_dir, "model.safetensors.index.json")
bi = os.path.join(base_dir, "model.safetensors.index.json")
conv, base = json.load(open(ci)), json.load(open(bi))
cw, bw = conv["weight_map"], base["weight_map"]

missing = sorted(set(bw) - set(cw))
if not missing:
    print(f"[ok] no missing keys; {conv_dir} already complete ({len(cw)} keys)")
    sys.exit(0)

NEW_SHARD = "model-90000-base-nonlm.safetensors"
out = os.path.join(conv_dir, NEW_SHARD)
if os.path.exists(out):
    print(f"[ok] extra shard already present in {conv_dir}; skipping")
    sys.exit(0)

print(f"[merge] {conv_dir}: adding {len(missing)} base keys "
      f"({sum(1 for k in missing if k.startswith('model.visual'))} visual, "
      f"{sum(1 for k in missing if k.startswith('mtp'))} mtp)")

by_shard = {}
for k in missing:
    by_shard.setdefault(bw[k], []).append(k)

tensors = {}
for shard, keys in sorted(by_shard.items()):
    with safe_open(os.path.join(base_dir, shard), framework="pt") as f:
        for k in keys:
            tensors[k] = f.get_tensor(k)

save_file(tensors, out, metadata={"format": "pt"})
added_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
for k in missing:
    cw[k] = NEW_SHARD
conv.setdefault("metadata", {})
conv["metadata"]["total_size"] = int(conv["metadata"].get("total_size", 0)) + int(added_bytes)
json.dump(conv, open(ci, "w"), indent=2)
print(f"[done] wrote {NEW_SHARD} ({added_bytes/1e9:.2f} GB); index now {len(cw)} keys "
      f"(was {len(cw)-len(missing)})")
# sanity
assert set(cw) == set(bw), f"key mismatch after merge: {len(set(cw)^set(bw))} differ"
print("[verify] merged key set == base key set")
