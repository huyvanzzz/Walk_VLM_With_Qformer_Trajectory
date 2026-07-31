def apply_mode_gated_trajectory_trainability(model, logger=None):
    if not getattr(model, "trajectory_enabled", False):
        return {}

    mode = getattr(model, "trajectory_fusion_mode", None)
    if mode != "concat":
        raise ValueError(
            f"Unsupported trajectory fusion mode for concat-only public repo: {mode}"
        )

    states = {
        "trajectory_backbone": True,
        "trajectory_cls_head": False,
        "trajectory_token_projector": True,
    }
    for module_name, should_train in states.items():
        module = getattr(model, module_name, None)
        if module is not None:
            module.requires_grad_(should_train)

    if logger is not None:
        logger.info(
            "Mode-gated trajectory trainability | mode=%s | backbone=%s | cls_head=%s | token_projector=%s",
            mode,
            states["trajectory_backbone"],
            states["trajectory_cls_head"],
            states["trajectory_token_projector"],
        )
    return states
