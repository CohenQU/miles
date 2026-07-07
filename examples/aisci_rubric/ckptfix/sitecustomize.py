# Fix for the Megatron distributed-optimizer dist-checkpoint SAVE/LOAD on qwen3_5_moe
# (Qwen3.6-35B-A3B) at TP8/PP2/CP4/EP8. Placed on the training job's PYTHONPATH by
# run_qwen3_6_35b_a3b_aisci_rubric.sh. Python imports `sitecustomize` at interpreter
# startup in the driver AND every ray actor, so this runs in the process that saves.
#
# BUG (megatron/core/optimizer/distrib_optimizer.py:~1673): each optimizer bucket
# ShardedTensor is built with global shape = gbuf_world_numel_unpadded but global
# offset = dp_rank * gbuf_local_numel, where gbuf_local_numel is the PADDED per-rank
# size. So the last DP rank's real shard extends past the unpadded global size by the
# trailing "padding to DP multiple". Megatron's code intends to discard that padding
# (see its docstring) but leaves a remainder (+64 elems here), so torch DCP's
# _validate_global_plan rejects the plan: "ValueError: Failed to validate global plan".
# Only optimizer buckets are affected (model weights save fine).
#
# FIX: right before dist_checkpointing.save()/.load(), walk the sharded_state_dict and
# clamp any 1-D ShardedTensor whose (global_offset + local_shape) exceeds global_shape,
# trimming the local data + local_shape to end exactly at global_shape. The trimmed tail
# is the trailing DP-multiple padding (region beyond the true unpadded param count), so
# NO optimizer state is lost and coverage becomes exact. Applied identically on save and
# load so the two stay consistent (we always resume at the same parallelism). Idempotent;
# no effect on runs that don't put this dir on PYTHONPATH (the 27B runs are untouched).
import os
import sys

_DEBUG = os.environ.get("AISCI_CKPT_DEBUG", "0") == "1"


def _clamp_sharded_state_dict(ssd):
    """Recursively clamp overshooting 1-D ShardedTensors in-place. Returns #clamped."""
    try:
        from megatron.core.dist_checkpointing.mapping import ShardedTensor
    except Exception:
        return 0
    n = 0
    stack = [ssd]
    seen = 0
    while stack:
        obj = stack.pop()
        seen += 1
        if seen > 5_000_000:  # runaway guard
            break
        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple)):
            stack.extend(obj)
            continue
        if isinstance(obj, ShardedTensor):
            try:
                gs = getattr(obj, "global_shape", None)
                go = getattr(obj, "global_offset", None)
                ls = getattr(obj, "local_shape", None)
                fr = getattr(obj, "flattened_range", None)
                # Only the plain 1-D flat optimizer buckets (flattened_range is None).
                if (fr is None and gs is not None and go is not None and ls is not None
                        and len(gs) == 1 and len(go) == 1 and len(ls) == 1):
                    end = go[0] + ls[0]
                    if end > gs[0]:
                        keep = max(0, gs[0] - go[0])
                        if keep < ls[0]:
                            data = getattr(obj, "data", None)
                            if data is not None:
                                obj.data = data[:keep]
                            obj.local_shape = (keep,)
                            n += 1
            except Exception as e:  # never break a save/load on the fix
                print("[CKPT_FIX] clamp error on a tensor: %r" % (e,), file=sys.stderr, flush=True)
    return n


def _patch_serialization(mod):
    if getattr(mod, "_aisci_ckpt_fix", False):
        return
    _orig_save = mod.save
    _orig_load = mod.load

    def save(sharded_state_dict, *a, **k):
        try:
            c = _clamp_sharded_state_dict(sharded_state_dict)
            if c and _DEBUG:
                print("[CKPT_FIX] save: clamped %d overshooting optimizer shard(s)" % c, flush=True)
        except Exception as e:
            print("[CKPT_FIX] save clamp failed: %r" % (e,), file=sys.stderr, flush=True)
        return _orig_save(sharded_state_dict, *a, **k)

    def load(sharded_state_dict, *a, **k):
        try:
            c = _clamp_sharded_state_dict(sharded_state_dict)
            if c and _DEBUG:
                print("[CKPT_FIX] load: clamped %d overshooting optimizer shard(s)" % c, flush=True)
        except Exception as e:
            print("[CKPT_FIX] load clamp failed: %r" % (e,), file=sys.stderr, flush=True)
        return _orig_load(sharded_state_dict, *a, **k)

    mod.save = save
    mod.load = load
    mod._aisci_ckpt_fix = True
    print("[CKPT_FIX] patched dist_checkpointing.save/.load (optimizer padding clamp)", flush=True)


# --- Lazy install: patch megatron serialization once it is first imported. ---
# (Avoids importing the heavy megatron.core at interpreter startup.)
import importlib.abc  # noqa: E402

_TARGET = "megatron.core.dist_checkpointing.serialization"


class _SerializationPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        try:
            idx = sys.meta_path.index(self)
        except ValueError:
            return None
        for finder in sys.meta_path[idx + 1:]:
            try:
                spec = finder.find_spec(name, path, target)
            except Exception:
                spec = None
            if spec is not None and spec.loader is not None:
                _orig_exec = spec.loader.exec_module

                def exec_module(module, _orig_exec=_orig_exec):
                    _orig_exec(module)
                    try:
                        _patch_serialization(module)
                    except Exception as e:
                        print("[CKPT_FIX] patch install failed: %r" % (e,), file=sys.stderr, flush=True)

                spec.loader.exec_module = exec_module
                return spec
        return None


if not any(isinstance(f, _SerializationPatchFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _SerializationPatchFinder())
    if _DEBUG:
        print("[CKPT_FIX] serialization patch finder installed", flush=True)


# --- Optional: with AISCI_CKPT_DEBUG=1, also log the exact tensors DCP validation
#     sees (should now report OK after the clamp). torch is safe to import at site time. ---
if _DEBUG:
    try:
        import operator
        from functools import reduce
        import torch.distributed.checkpoint.default_planner as _dp
        from torch.distributed.checkpoint.metadata import BytesStorageMetadata

        _orig_validate = _dp._validate_global_plan

        def _validate_global_plan(global_plan, metadata):
            bad = []
            for key, value in metadata.state_dict_metadata.items():
                if isinstance(value, BytesStorageMetadata) or len(value.size) == 0:
                    continue
                vol = 0
                for i, c0 in enumerate(value.chunks):
                    if not _dp._check_box_bounds(value.size, c0):
                        bad.append((key, "OOB", tuple(value.size), tuple(c0.offsets), tuple(c0.sizes)))
                    vol += reduce(operator.mul, c0.sizes, 1)
                    for c1 in value.chunks[i + 1:]:
                        if _dp._check_box_overlap(c0, c1):
                            bad.append((key, "OVERLAP", tuple(c0.offsets), tuple(c0.sizes)))
                tv = reduce(operator.mul, value.size, 1)
                if len(global_plan) > 1 and vol != tv:
                    bad.append((key, "COVER", tuple(value.size), "tvol=%d" % tv, "cvol=%d" % vol))
            if bad:
                print("[CKPT_DEBUG] validation FAILED for %d tensor(s):" % len(bad), flush=True)
                for b in bad[:60]:
                    print("[CKPT_DEBUG]  ", *b, flush=True)
            else:
                print("[CKPT_DEBUG] validation OK (plan valid after clamp)", flush=True)
            return _orig_validate(global_plan, metadata)

        _dp._validate_global_plan = _validate_global_plan
        print("[CKPT_DEBUG] patched torch DCP _validate_global_plan (logging)", flush=True)
    except Exception as _e:
        print("[CKPT_DEBUG] validate patch failed: %r" % (_e,), file=sys.stderr, flush=True)
