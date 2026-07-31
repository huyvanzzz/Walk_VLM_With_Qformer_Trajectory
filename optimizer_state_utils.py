import copy

import torch


def clone_optimizer_state_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_optimizer_state_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_optimizer_state_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_optimizer_state_value(v) for v in value)
    return copy.deepcopy(value)


def clone_optimizer_state_dict(optimizer_state_dict):
    return clone_optimizer_state_value(optimizer_state_dict)


def sanitize_optimizer_state_dict(optimizer_state_dict):
    state = optimizer_state_dict.get("state", {})
    converted = 0
    for param_state in state.values():
        if not isinstance(param_state, dict):
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = param_state.get(key)
            if torch.is_tensor(tensor) and tensor.dtype != torch.float32:
                param_state[key] = tensor.float()
                converted += 1
    param_groups = optimizer_state_dict.get("param_groups", [])
    overridden_groups = 0
    for group in param_groups:
        changed = False
        if group.get("foreach", None) is not False:
            group["foreach"] = False
            changed = True
        if "fused" in group and group.get("fused", None) is not False:
            group["fused"] = False
            changed = True
        if changed:
            overridden_groups += 1
    return optimizer_state_dict, converted, overridden_groups


def export_sanitized_optimizer_state_dict(optimizer):
    optimizer_state_dict = clone_optimizer_state_dict(optimizer.state_dict())
    return sanitize_optimizer_state_dict(optimizer_state_dict)


def move_optimizer_state_to_param_device(optimizer):
    moved = 0
    for param, param_state in optimizer.state.items():
        if not isinstance(param_state, dict):
            continue
        param_device = param.device
        for key, value in param_state.items():
            if torch.is_tensor(value) and value.device != param_device:
                param_state[key] = value.to(device=param_device)
                moved += 1
    return moved


def enforce_safe_optimizer_param_groups(optimizer):
    overridden_groups = 0
    for group in optimizer.param_groups:
        changed = False
        if group.get("foreach", None) is not False:
            group["foreach"] = False
            changed = True
        if "fused" in group and group.get("fused", None) is not False:
            group["fused"] = False
            changed = True
        if changed:
            overridden_groups += 1
    return overridden_groups


def count_optimizer_state_tensors_on_cpu(optimizer):
    count = 0
    for param_state in optimizer.state.values():
        if not isinstance(param_state, dict):
            continue
        for value in param_state.values():
            if torch.is_tensor(value) and value.device.type == "cpu":
                count += 1
    return count
