import json
import os
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from huggingface_hub.utils import EntryNotFoundError
from transformers import InstructBlipConfig, InstructBlipProcessor, InstructBlipQFormerModel
from huggingface_hub import hf_hub_download
from trajectory_branch import (
    attach_trajectory_branch,
    build_trajectory_features,
    build_trajectory_tokens_base,
)


BRIDGE_WEIGHTS_NAME = "qformer_bridge.safetensors"
BRIDGE_CONFIG_NAME = "qformer_bridge_config.json"


def qformer_enabled(config: Dict) -> bool:
    q_cfg = config.get("qformer", config.get("model", {}).get("qformer", {}))
    return bool(q_cfg.get("enabled", False))


def get_qformer_config(config: Dict) -> Dict:
    q_cfg = dict(config.get("qformer", config.get("model", {}).get("qformer", {})))
    q_cfg.setdefault("enabled", False)
    q_cfg.setdefault("source_model", "Salesforce/instructblip-flan-t5-xl")
    q_cfg.setdefault("cache_dir", "./qformer_cache")
    q_cfg.setdefault("num_query_tokens", 32)
    q_cfg.setdefault("freeze_qformer", True)
    q_cfg.setdefault("freeze_mlp1", True)
    q_cfg.setdefault("prompt_aware", True)
    q_cfg.setdefault("max_text_length", 128)
    return q_cfg


def _module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def _download_qformer_state(source_model: str, cache_dir: str) -> Dict[str, torch.Tensor]:
    use_safetensors = True
    single_file = None
    try:
        index_path = hf_hub_download(
            repo_id=source_model,
            filename="model.safetensors.index.json",
            cache_dir=cache_dir,
        )
    except EntryNotFoundError:
        use_safetensors = False
        try:
            index_path = hf_hub_download(
                repo_id=source_model,
                filename="pytorch_model.bin.index.json",
                cache_dir=cache_dir,
            )
        except EntryNotFoundError:
            try:
                single_file = hf_hub_download(
                    repo_id=source_model,
                    filename="model.safetensors",
                    cache_dir=cache_dir,
                )
                use_safetensors = True
            except EntryNotFoundError:
                single_file = hf_hub_download(
                    repo_id=source_model,
                    filename="pytorch_model.bin",
                    cache_dir=cache_dir,
                )

    if single_file is not None:
        state = load_file(single_file, device="cpu") if use_safetensors else torch.load(single_file, map_location="cpu")
        return {
            name: tensor
            for name, tensor in state.items()
            if name.startswith(("qformer.", "query_tokens"))
        }

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    needed_prefixes = ("qformer.", "query_tokens")
    weight_map = index["weight_map"]
    shard_names = sorted(
        {
            shard
            for name, shard in weight_map.items()
            if name.startswith(needed_prefixes)
        }
    )
    if not shard_names:
        raise RuntimeError(f"No Q-Former weights found in {source_model}.")

    qformer_state = {}
    for shard_name in shard_names:
        shard_path = hf_hub_download(
            repo_id=source_model,
            filename=shard_name,
            cache_dir=cache_dir,
        )
        shard = load_file(shard_path, device="cpu") if use_safetensors else torch.load(shard_path, map_location="cpu")
        for name, tensor in shard.items():
            if name.startswith(needed_prefixes):
                qformer_state[name] = tensor
    return qformer_state


def _load_qformer_from_source(source_model: str, cache_dir: str):
    blip_config = InstructBlipConfig.from_pretrained(source_model, cache_dir=cache_dir)
    qformer = InstructBlipQFormerModel(blip_config.qformer_config)
    query_tokens = nn.Parameter(
        torch.zeros(
            1,
            blip_config.num_query_tokens,
            blip_config.qformer_config.hidden_size,
        )
    )

    state = _download_qformer_state(source_model, cache_dir)
    qformer_state = {
        name[len("qformer.") :]: tensor
        for name, tensor in state.items()
        if name.startswith("qformer.")
    }
    missing, unexpected = qformer.load_state_dict(qformer_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Q-Former keys: {unexpected[:5]}")
    if missing:
        print(f"[QFormerBridge] Warning: missing Q-Former keys: {missing[:5]}")
    if "query_tokens" not in state:
        raise RuntimeError("query_tokens not found in Q-Former checkpoint.")
    query_tokens.data.copy_(state["query_tokens"])

    processor = InstructBlipProcessor.from_pretrained(source_model, cache_dir=cache_dir)
    return qformer, query_tokens, processor.qformer_tokenizer, blip_config


def _extract_vit_tokens(model, pixel_values):
    if model.select_layer == -1:
        vit_embeds = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    else:
        vit_embeds = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[model.select_layer]
    vit_embeds = vit_embeds[:, 1:, :]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = model.pixel_shuffle(vit_embeds, scale_factor=model.downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    return vit_embeds


def _ensure_bridge_device(model, reference: torch.Tensor):
    device = reference.device
    reference_dtype = reference.dtype

    qformer = getattr(model, "qformer", None)
    if qformer is not None:
        qformer.to(device=device, dtype=reference_dtype)

    for module_name in ("qformer_input_proj", "qformer_to_mlp1_proj"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.to(device=device, dtype=torch.float32)

    for module_name in ("trajectory_backbone", "trajectory_token_projector"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.to(device=device, dtype=torch.float32)

    model.qformer_query_tokens.data = model.qformer_query_tokens.data.to(device=device, dtype=reference_dtype)


def align_qformer_bridge_runtime(model):
    if not getattr(model, "qformer_enabled", False):
        return False

    reference_param = None
    for candidate_name in ("mlp1", "vision_model"):
        module = getattr(model, candidate_name, None)
        if module is None:
            continue
        try:
            reference_param = next(module.parameters())
            break
        except StopIteration:
            continue

    if reference_param is None:
        return False

    _ensure_bridge_device(model, reference_param.data)
    return True


def _extract_feature_with_qformer(self, pixel_values):
    vit_embeds = _extract_vit_tokens(self, pixel_values)
    _ensure_bridge_device(self, vit_embeds)

    proj_in_dtype = next(self.qformer_input_proj.parameters()).dtype
    encoder_hidden_states = self.qformer_input_proj(vit_embeds.to(proj_in_dtype))
    encoder_attention_mask = torch.ones(
        encoder_hidden_states.size()[:-1],
        dtype=torch.long,
        device=encoder_hidden_states.device,
    )

    qformer_input_ids = getattr(self, "_qformer_input_ids", None)
    qformer_attention_mask = getattr(self, "_qformer_attention_mask", None)
    if qformer_input_ids is None or qformer_attention_mask is None:
        qformer_input_ids, qformer_attention_mask = self.encode_qformer_texts(
            ["" for _ in range(pixel_values.shape[0])],
            device=encoder_hidden_states.device,
        )
    else:
        qformer_input_ids = qformer_input_ids.to(encoder_hidden_states.device)
        qformer_attention_mask = qformer_attention_mask.to(encoder_hidden_states.device)

    if qformer_input_ids.shape[0] != encoder_hidden_states.shape[0]:
        raise ValueError(
            "Q-Former text batch must match image/tile batch: "
            f"{qformer_input_ids.shape[0]} vs {encoder_hidden_states.shape[0]}"
        )

    query_tokens = self.qformer_query_tokens.expand(encoder_hidden_states.shape[0], -1, -1)
    query_attention_mask = torch.ones(
        query_tokens.size()[:-1],
        dtype=torch.long,
        device=encoder_hidden_states.device,
    )
    qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
    qformer_dtype = next(self.qformer.parameters()).dtype
    encoder_hidden_states = encoder_hidden_states.to(qformer_dtype)
    query_tokens = query_tokens.to(qformer_dtype)
    query_outputs = self.qformer(
        input_ids=qformer_input_ids,
        attention_mask=qformer_attention_mask,
        query_embeds=query_tokens,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        return_dict=True,
    )
    query_output = query_outputs[0][:, : query_tokens.size(1), :]
    proj_out_dtype = next(self.qformer_to_mlp1_proj.parameters()).dtype
    mlp1_inputs = self.qformer_to_mlp1_proj(query_output.to(proj_out_dtype))
    mlp1_dtype = next(self.mlp1.parameters()).dtype
    mlp1_inputs = mlp1_inputs.to(mlp1_dtype)
    visual_tokens = self.mlp1(mlp1_inputs)
    if getattr(self, "trajectory_enabled", False):
        if self.trajectory_fusion_mode != "concat":
            raise ValueError(
                f"Unsupported trajectory fusion_mode for concat-only public repo: {self.trajectory_fusion_mode}"
            )
        traj_tokens = build_trajectory_features(
            self,
            batch_size=visual_tokens.shape[0],
            device=visual_tokens.device,
        ).to(visual_tokens.dtype)
        visual_tokens = torch.cat([visual_tokens, traj_tokens], dim=1)
    return visual_tokens


def _encode_qformer_texts(self, texts: List[str], device: Optional[torch.device] = None):
    encoded = self.qformer_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=self.qformer_max_text_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _set_qformer_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    self._qformer_input_ids = input_ids
    self._qformer_attention_mask = attention_mask


def _clear_qformer_text(self):
    self._qformer_input_ids = None
    self._qformer_attention_mask = None


def _bridge_state_dict(model) -> Dict[str, torch.Tensor]:
    state = {}
    for module_name in ("qformer_input_proj", "qformer_to_mlp1_proj"):
        module = getattr(model, module_name, None)
        if module is not None:
            for key, value in module.state_dict().items():
                state[f"{module_name}.{key}"] = value.detach().cpu()
    return state


def save_qformer_bridge(model, output_dir: str):
    if not getattr(model, "qformer_enabled", False):
        return
    os.makedirs(output_dir, exist_ok=True)
    state = _bridge_state_dict(model)
    save_file(state, os.path.join(output_dir, BRIDGE_WEIGHTS_NAME))
    metadata = {
        "enabled": True,
        "source_model": model.qformer_source_model,
        "num_query_tokens": getattr(model, "qformer_num_query_tokens", model.num_image_token),
        "prompt_aware": True,
        "bridge_mode": "prompt_aware_preproj_mlp1",
        "stage": getattr(model, "pretrain_stage", getattr(model, "training_stage", None)),
        "movement_enabled": getattr(model, "pretrain_movement_enabled", None),
        "pretrain_data_source": getattr(model, "pretrain_data_source", None),
        "question_format_version": getattr(model, "question_format_version", None),
    }
    with open(os.path.join(output_dir, BRIDGE_CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def read_qformer_bridge_metadata(checkpoint_dir: str) -> Dict:
    config_path = os.path.join(checkpoint_dir, BRIDGE_CONFIG_NAME)
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qformer_bridge(model, checkpoint_dir: str, strict: bool = True):
    weights_path = os.path.join(checkpoint_dir, BRIDGE_WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        if strict:
            raise FileNotFoundError(f"Q-Former bridge weights not found: {weights_path}")
        return False
    state = load_file(weights_path, device="cpu")
    for module_name in ("qformer_input_proj", "qformer_to_mlp1_proj"):
        module_state = {
            key[len(module_name) + 1 :]: value
            for key, value in state.items()
            if key.startswith(module_name + ".")
        }
        getattr(model, module_name).load_state_dict(module_state, strict=True)
    return True


def attach_qformer_bridge(model, config: Dict, logger=None):
    q_cfg = get_qformer_config(config)
    if not q_cfg["enabled"]:
        model.qformer_enabled = False
        return model

    qformer, query_tokens, qformer_tokenizer, blip_config = _load_qformer_from_source(
        q_cfg["source_model"],
        q_cfg["cache_dir"],
    )

    vit_hidden_size = model.config.vision_config.hidden_size
    llm_hidden_size = model.config.llm_config.hidden_size
    downsample_ratio = model.config.downsample_ratio
    pixel_shuffle_dim = vit_hidden_size * int(1 / downsample_ratio) ** 2
    qformer_encoder_dim = blip_config.qformer_config.encoder_hidden_size
    qformer_hidden_size = blip_config.qformer_config.hidden_size
    num_query_tokens = int(q_cfg["num_query_tokens"])

    if query_tokens.shape[1] < num_query_tokens:
        raise ValueError(
            f"Requested {num_query_tokens} query tokens, checkpoint has {query_tokens.shape[1]}"
        )
    query_tokens = nn.Parameter(query_tokens[:, :num_query_tokens, :].clone())

    model.qformer_enabled = True
    model.qformer_source_model = q_cfg["source_model"]
    model.qformer_max_text_length = int(q_cfg["max_text_length"])
    model.qformer_tokenizer = qformer_tokenizer
    model.qformer = qformer
    model.qformer_query_tokens = query_tokens
    model.qformer_input_proj = nn.Sequential(
        nn.LayerNorm(pixel_shuffle_dim),
        nn.Linear(pixel_shuffle_dim, qformer_encoder_dim),
    ).to(dtype=torch.float32)
    model.qformer_to_mlp1_proj = nn.Sequential(
        nn.LayerNorm(qformer_hidden_size),
        nn.Linear(qformer_hidden_size, pixel_shuffle_dim),
    ).to(dtype=torch.float32)
    model.qformer_num_query_tokens = num_query_tokens
    model.num_image_token = num_query_tokens

    if q_cfg["freeze_qformer"]:
        model.qformer.requires_grad_(False)
        model.qformer_query_tokens.requires_grad_(False)
    if q_cfg["freeze_mlp1"]:
        model.mlp1.requires_grad_(False)

    model.extract_feature = MethodType(_extract_feature_with_qformer, model)
    model.encode_qformer_texts = MethodType(_encode_qformer_texts, model)
    model.set_qformer_text = MethodType(_set_qformer_text, model)
    model.clear_qformer_text = MethodType(_clear_qformer_text, model)

    if logger:
        logger.info(
            "Attached prompt-aware Q-Former bridge: "
            f"pixel_shuffle_dim={pixel_shuffle_dim}, "
            f"qformer_encoder_dim={qformer_encoder_dim}, "
            f"qformer_hidden_size={qformer_hidden_size}, "
            f"num_query_tokens={num_query_tokens}, "
            f"llm_hidden_size={llm_hidden_size}"
        )
    attach_trajectory_branch(
        model,
        config,
        pixel_shuffle_dim=pixel_shuffle_dim,
        llm_hidden_size=llm_hidden_size,
        logger=logger,
    )
    return model


def trainable_parameter_summary(model) -> List[str]:
    rows = []
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total += count
            rows.append(f"{name}: {count:,}")
    rows.append(f"TOTAL_TRAINABLE: {total:,}")
    return rows
