import argparse
import json
import os
from collections import Counter

import torch
from huggingface_hub import snapshot_download


def resolve_checkpoint_path(checkpoint: str) -> str:
    if os.path.exists(checkpoint):
        return checkpoint
    print(f"[INFO] Checkpoint is not a local path. Downloading from Hugging Face: {checkpoint}")
    return snapshot_download(
        repo_id=checkpoint,
        allow_patterns=[
            "optimizer.pt",
            "scheduler.pt",
            "adapter_config.json",
            "qformer_bridge.safetensors",
            "qformer_bridge_config.json",
        ],
    )


def tensor_summary(tensor: torch.Tensor) -> dict:
    return {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
    }


def inspect_optimizer_state(optimizer_state: dict, sample_limit: int) -> None:
    state = optimizer_state.get("state", {})
    param_groups = optimizer_state.get("param_groups", [])

    print("=== OPTIMIZER SUMMARY ===")
    print(f"param_groups: {len(param_groups)}")
    print(f"state entries: {len(state)}")

    group_keys = Counter()
    for group in param_groups:
        group_keys.update(group.keys())
    print(f"param_group keys: {sorted(group_keys.keys())}")

    tensor_dtype_counter = Counter()
    tensor_device_counter = Counter()
    state_key_counter = Counter()

    sampled = []
    for param_id, param_state in state.items():
        if not isinstance(param_state, dict):
            continue
        sample_item = {"param_id": str(param_id), "state_keys": sorted(param_state.keys()), "tensors": {}}
        for key, value in param_state.items():
            state_key_counter.update([key])
            if torch.is_tensor(value):
                tensor_dtype_counter.update([str(value.dtype)])
                tensor_device_counter.update([str(value.device)])
                if len(sampled) < sample_limit:
                    sample_item["tensors"][key] = tensor_summary(value)
            else:
                if len(sampled) < sample_limit:
                    sample_item["tensors"][key] = {
                        "type": type(value).__name__,
                        "value": value,
                    }
        if len(sampled) < sample_limit:
            sampled.append(sample_item)

    print(f"state key frequencies: {dict(state_key_counter)}")
    print(f"tensor dtypes: {dict(tensor_dtype_counter)}")
    print(f"tensor devices: {dict(tensor_device_counter)}")
    print()
    print("=== SAMPLE STATE ENTRIES ===")
    print(json.dumps(sampled, indent=2, ensure_ascii=False))


def inspect_scheduler_state(scheduler_state: dict) -> None:
    print()
    print("=== SCHEDULER SUMMARY ===")
    print(f"keys: {sorted(scheduler_state.keys())}")
    compact = {
        key: scheduler_state[key]
        for key in ["last_epoch", "_step_count", "base_lrs"]
        if key in scheduler_state
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Inspect optimizer/scheduler checkpoint state.")
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint dir or Hugging Face repo id.")
    parser.add_argument("--sample_limit", type=int, default=5, help="How many optimizer state entries to print.")
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint_path(args.checkpoint)
    print(f"[INFO] Resolved checkpoint dir: {checkpoint_dir}")

    opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
    sch_path = os.path.join(checkpoint_dir, "scheduler.pt")

    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"optimizer.pt not found in: {checkpoint_dir}")

    optimizer_state = torch.load(opt_path, map_location="cpu")
    inspect_optimizer_state(optimizer_state, sample_limit=args.sample_limit)

    if os.path.exists(sch_path):
        scheduler_state = torch.load(sch_path, map_location="cpu")
        inspect_scheduler_state(scheduler_state)
    else:
        print()
        print("[INFO] scheduler.pt not found.")


if __name__ == "__main__":
    main()
