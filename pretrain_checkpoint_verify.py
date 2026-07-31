from __future__ import annotations

import os
from typing import Dict, List

import torch
from safetensors.torch import load_file

from qformer_bridge import BRIDGE_WEIGHTS_NAME
from trajectory_branch import TRAJECTORY_BRANCH_WEIGHTS_NAME


def _collect_module_keys(state: Dict[str, torch.Tensor], prefix: str) -> List[str]:
    return sorted(key for key in state.keys() if key.startswith(prefix + "."))


def _compare_selected_keys(model, checkpoint_state: Dict[str, torch.Tensor], module_names: List[str], max_keys: int = 3):
    checked = []
    for module_name in module_names:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        module_keys = _collect_module_keys(checkpoint_state, module_name)
        for full_key in module_keys[:max_keys]:
            param_key = full_key[len(module_name) + 1 :]
            model_tensor = module.state_dict()[param_key].detach().cpu()
            checkpoint_tensor = checkpoint_state[full_key].detach().cpu()
            matched = torch.equal(model_tensor, checkpoint_tensor)
            checked.append(
                {
                    "key": full_key,
                    "matched": bool(matched),
                    "model_norm": round(float(model_tensor.norm().item()), 6),
                    "checkpoint_norm": round(float(checkpoint_tensor.norm().item()), 6),
                }
            )
    return {
        "all_matched": bool(checked) and all(item["matched"] for item in checked),
        "checked_keys": checked,
    }


def verify_loaded_pretrain_modules(model, checkpoint_dir: str):
    bridge_path = os.path.join(checkpoint_dir, BRIDGE_WEIGHTS_NAME)
    trajectory_path = os.path.join(checkpoint_dir, TRAJECTORY_BRANCH_WEIGHTS_NAME)
    if not os.path.exists(bridge_path):
        raise FileNotFoundError(f"Missing bridge checkpoint for verification: {bridge_path}")
    if not os.path.exists(trajectory_path):
        raise FileNotFoundError(f"Missing trajectory checkpoint for verification: {trajectory_path}")

    bridge_state = load_file(bridge_path, device="cpu")
    trajectory_state = load_file(trajectory_path, device="cpu")
    return {
        "bridge": _compare_selected_keys(
            model,
            bridge_state,
            ["qformer_input_proj", "qformer_to_mlp1_proj"],
        ),
        "trajectory": _compare_selected_keys(
            model,
            trajectory_state,
            ["trajectory_backbone", "trajectory_token_projector"],
        ),
    }
