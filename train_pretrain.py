import argparse
import datetime
import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from math import floor
from typing import Dict

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from logutil import get_logger, init_logger
from model.conversation import get_conv_template
from optimizer_state_utils import enforce_safe_optimizer_param_groups
from pretrain_dataset import DEFAULT_FRAME_INDEX_PATH, build_pretrain_datasets
from qformer_bridge import (
    align_qformer_bridge_runtime,
    attach_qformer_bridge,
    load_qformer_bridge,
    qformer_enabled,
    read_qformer_bridge_metadata,
    save_qformer_bridge,
    trainable_parameter_summary,
)
from trajectory_branch import (
    build_trajectory_source_from_config,
    load_trajectory_branch,
    read_trajectory_branch_metadata,
    save_trajectory_branch,
    trajectory_enabled,
)
from trajectory_trainability import apply_mode_gated_trajectory_trainability


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SYSTEM_MESSAGE = "You are a navigation assistant for visually impaired users."
EARLY_STOPPING_STATE_FILENAME = "early_stopping_state.json"


@dataclass
class EarlyStoppingState:
    patience: int
    min_delta: float
    best_val_loss: float | None = None
    best_epoch: int | None = None
    num_bad_epochs: int = 0

    def update(self, epoch: int, val_loss: float):
        improved = self.best_val_loss is None or val_loss < (self.best_val_loss - self.min_delta)
        if improved:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return True, False

        self.num_bad_epochs += 1
        should_stop = self.num_bad_epochs >= self.patience
        return False, should_stop


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PretrainCollaterFn:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.log_token_stats = False
        self.token_log_remaining = 0

    def __call__(self, batch):
        label_ids_batch = []
        input_ids_batch = []
        attention_mask_batch = []
        pixel_values_batch = []
        qformer_texts = []
        trajectory_label_ids_batch = []
        trajectory_direction_ids_batch = []
        trajectory_numeric_feats_batch = []
        trajectory_object_mask_batch = []
        samples_batch = []

        for sample in batch:
            question = sample["question"]
            answer = sample["answer"]
            pixel_values = sample["pixel_values"]
            samples_batch.append(sample)

            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            num_patches_list = [pv.shape[0] for pv in pixel_values]
            total_tiles = sum(num_patches_list)
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace("<image>", image_tokens, 1)

            input_ids = self.tokenizer.encode(query, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

            if self.log_token_stats and self.token_log_remaining != 0:
                total_image_tokens_in_sample = total_tiles * self.model.num_image_token
                total_sequence_length = len(input_ids) + len(answer_ids) + 1
                logger = get_logger()
                logger.info(
                    "[INFO] Pretrain image token stats | frames=%s | tiles_per_frame=%s | query_tokens_per_tile=%s | total_image_tokens=%s",
                    len(pixel_values),
                    num_patches_list,
                    self.model.num_image_token,
                    total_image_tokens_in_sample,
                )
                logger.info(
                    "[INFO] Pretrain text tokens | input=%s | answer=%s | total=%s | total_input_tokens=%s",
                    len(input_ids),
                    len(answer_ids),
                    total_sequence_length,
                    len(input_ids),
                )
                if self.token_log_remaining > 0:
                    self.token_log_remaining -= 1

            label_ids = [-100] * len(input_ids) + answer_ids + [eos_token_id]
            input_ids = input_ids + answer_ids + [eos_token_id]
            attention_mask = [1] * len(input_ids)

            label_ids_batch.append(label_ids)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pixel_values_batch.append(torch.cat(pixel_values, dim=0))

            if getattr(self.model, "qformer_enabled", False):
                qformer_text = sample.get("qformer_text", question.replace("<image>", "").strip())
                qformer_texts.extend([qformer_text] * total_tiles)

            if getattr(self.model, "trajectory_enabled", False):
                for _ in range(total_tiles):
                    trajectory_label_ids_batch.append(torch.as_tensor(sample["trajectory_label_ids"], dtype=torch.long))
                    trajectory_direction_ids_batch.append(torch.as_tensor(sample["trajectory_direction_ids"], dtype=torch.long))
                    trajectory_numeric_feats_batch.append(torch.as_tensor(sample["trajectory_numeric_feats"], dtype=torch.float32))
                    trajectory_object_mask_batch.append(torch.as_tensor(sample["trajectory_object_mask"], dtype=torch.long))

        input_ids_tensor = _maybe_pad(input_ids_batch, eot_token_id)
        label_ids_tensor = _maybe_pad(label_ids_batch, -100)
        attention_mask_tensor = _maybe_pad(attention_mask_batch, 0)
        pixel_values_tensor = torch.cat(pixel_values_batch)

        qformer_inputs = None
        if getattr(self.model, "qformer_enabled", False):
            qformer_inputs = self.model.encode_qformer_texts(qformer_texts)

        trajectory_inputs = None
        if getattr(self.model, "trajectory_enabled", False):
            trajectory_inputs = (
                torch.stack(trajectory_label_ids_batch, dim=0),
                torch.stack(trajectory_direction_ids_batch, dim=0),
                torch.stack(trajectory_numeric_feats_batch, dim=0),
                torch.stack(trajectory_object_mask_batch, dim=0),
            )

        return (
            input_ids_tensor,
            label_ids_tensor,
            attention_mask_tensor,
            pixel_values_tensor,
            qformer_inputs,
            trajectory_inputs,
            samples_batch,
        )


def _maybe_pad(inner_lists, padding_value):
    tensor_list = [torch.tensor(inner_list, dtype=torch.long) for inner_list in inner_lists]
    return pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)


def forward_pretrain_batch(model, batch, device: torch.device):
    input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, qformer_inputs, trajectory_inputs, _ = batch
    input_ids_batch = input_ids_batch.to(device)
    label_ids_batch = label_ids_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)
    pixel_values_batch = pixel_values_batch.to(
        torch.bfloat16 if device.type == "cuda" else torch.float32
    ).to(device)
    image_flags_batch = torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long, device=device)

    if getattr(model, "qformer_enabled", False) and qformer_inputs is not None:
        model.set_qformer_text(qformer_inputs[0].to(device), qformer_inputs[1].to(device))
    if getattr(model, "trajectory_enabled", False) and trajectory_inputs is not None:
        model.set_trajectory_inputs(
            trajectory_inputs[0].to(device),
            trajectory_inputs[1].to(device),
            trajectory_inputs[2].to(device),
            trajectory_inputs[3].to(device),
        )

    try:
        outputs = model(
            input_ids=input_ids_batch,
            pixel_values=pixel_values_batch,
            labels=label_ids_batch,
            image_flags=image_flags_batch,
            return_dict=True,
        )
    finally:
        if getattr(model, "qformer_enabled", False):
            model.clear_qformer_text()
        if getattr(model, "trajectory_enabled", False):
            model.clear_trajectory_inputs()
    return outputs.loss


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain InternVL-QFormer trajectory branch with generation loss.")
    parser.add_argument("--config", type=str, required=True, help="Path to pretrain YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional pretrain checkpoint dir or HF repo id to resume pretrain.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Alias for --checkpoint when resuming pretrain.",
    )
    parser.add_argument("--start_epoch", type=int, default=None, help="Zero-based epoch index to resume from.")
    parser.add_argument("--start_step", type=int, default=None, help="Batch step inside the resume epoch.")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_checkpoint_path(checkpoint: str | None, logger=None):
    if not checkpoint:
        return None
    if os.path.exists(checkpoint):
        if logger:
            logger.info(f"Using local pretrain checkpoint path: {checkpoint}")
        return checkpoint
    if logger:
        logger.info(f"Pretrain checkpoint is not a local path. Downloading from Hugging Face: {checkpoint}")
    return snapshot_download(
        repo_id=checkpoint,
        allow_patterns=[
            "qformer_bridge.safetensors",
            "qformer_bridge_config.json",
            "trajectory_branch.safetensors",
            "trajectory_branch_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "training_state.json",
            EARLY_STOPPING_STATE_FILENAME,
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )


def infer_resume_position(checkpoint_dir):
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return int(state.get("next_epoch", 0)), int(state.get("next_step", 0))

    name = os.path.basename(os.path.normpath(checkpoint_dir))
    step_match = re.fullmatch(r"epoch_(\d+)_step_(\d+)", name)
    if step_match:
        epoch_num = int(step_match.group(1))
        step = int(step_match.group(2))
        return max(epoch_num - 1, 0), step

    epoch_match = re.fullmatch(r"epoch_(\d+)", name)
    if epoch_match:
        epoch_num = int(epoch_match.group(1))
        return epoch_num, 0

    return None, None


def resolve_resume_config(args, logger):
    requested_checkpoint = args.resume_checkpoint or args.checkpoint
    checkpoint = resolve_checkpoint_path(requested_checkpoint, logger=logger)
    start_epoch = args.start_epoch
    start_step = args.start_step
    inferred_epoch, inferred_step = infer_resume_position(checkpoint) if checkpoint else (None, None)
    if checkpoint and logger:
        logger.info("Resolved pretrain resume checkpoint path: %s", checkpoint)
        logger.info(
            "Resume state discovery | training_state.json=%s | inferred_epoch=%s | inferred_step=%s",
            os.path.exists(os.path.join(checkpoint, "training_state.json")),
            inferred_epoch if inferred_epoch is not None else "missing",
            inferred_step if inferred_step is not None else "missing",
        )
    if start_epoch is None:
        start_epoch = inferred_epoch if inferred_epoch is not None else 0
    if start_step is None:
        start_step = inferred_step if inferred_step is not None else 0
    if checkpoint and logger:
        logger.info("Final pretrain resume position | start_epoch=%s | start_step=%s", start_epoch, start_step)
    return checkpoint, int(start_epoch or 0), int(start_step or 0)


def build_output_dir(config: Dict) -> str:
    base_out_dir = config["training"]["output_dir"]
    mode = str(config["trajectory"]["fusion_mode"])
    movement_flag = "movement_on" if bool(config["pretrain"]["movement_enabled"]) else "movement_off"
    run_name = f"stage_pretrain__{mode}__{movement_flag}__{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    output_dir = os.path.join(base_out_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_model_and_tokenizer(config: Dict, logger):
    model_name_or_path = config["model"]["name"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["quantization"]["enabled"],
        bnb_4bit_compute_dtype=torch.bfloat16 if config["model"]["quantization"]["compute_dtype"] == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=config["model"]["quantization"]["double_quant"],
        bnb_4bit_quant_type=config["model"]["quantization"]["type"],
    )

    logger.info("Loading model %s in pretrain mode...", model_name_or_path)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    model.config.use_cache = False
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.system_message = SYSTEM_MESSAGE

    if config["model"]["vision"]["freeze_encoder"]:
        model.vision_model.requires_grad_(False)
    if not qformer_enabled(config):
        raise ValueError("Pretrain core only supports InternVL-QFormer with qformer.enabled=true.")

    attach_qformer_bridge(model, config, logger=logger)
    align_qformer_bridge_runtime(model)

    if not trajectory_enabled(config):
        raise ValueError("Pretrain core requires trajectory.enabled=true.")

    model.pretrain_stage = "pretrain"
    model.training_stage = "pretrain"
    model.pretrain_data_source = config["pretrain"]["question_train_file"]
    model.question_format_version = "v1_qa_question_answer"
    model.pretrain_movement_enabled = bool(config["pretrain"]["movement_enabled"])

    model.language_model.requires_grad_(False)
    _freeze_modules_for_pretrain(model)
    model.train()
    return model, tokenizer


def _freeze_modules_for_pretrain(model):
    if hasattr(model, "qformer"):
        model.qformer.requires_grad_(False)
    if hasattr(model, "qformer_query_tokens"):
        model.qformer_query_tokens.requires_grad_(False)
    if hasattr(model, "mlp1"):
        model.mlp1.requires_grad_(False)
    if hasattr(model, "vision_model"):
        model.vision_model.requires_grad_(False)

    for module_name in (
        "qformer_input_proj",
        "qformer_to_mlp1_proj",
        "trajectory_backbone",
        "trajectory_token_projector",
    ):
        module = getattr(model, module_name, None)
        if module is not None:
            module.requires_grad_(True)
    apply_mode_gated_trajectory_trainability(model)


def build_dataloaders(config: Dict, model, tokenizer):
    pretrain_cfg = config["pretrain"]
    train_dataset, val_dataset = build_pretrain_datasets(
        question_train_file=pretrain_cfg["question_train_file"],
        frame_index_path=pretrain_cfg.get("frame_index_file", DEFAULT_FRAME_INDEX_PATH),
        trajectory_source=build_trajectory_source_from_config(config),
        val_split_ratio=float(pretrain_cfg["val_split_ratio"]),
        val_split_seed=int(pretrain_cfg["val_split_seed"]),
        movement_enabled=bool(pretrain_cfg["movement_enabled"]),
    )

    collate_fn = PretrainCollaterFn(tokenizer, model)
    collate_fn.log_token_stats = bool(config["training"].get("log_token_stats", False))
    collate_fn.token_log_remaining = int(config["training"].get("token_log_batches", 0))

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["hardware"].get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def resolve_warmup_steps(config: Dict, total_training_steps: int) -> int:
    training_cfg = config["training"]
    if "warmup_ratio" in training_cfg:
        warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
        warmup_min_steps = int(training_cfg.get("warmup_min_steps", 0))
        warmup_max_steps = int(training_cfg.get("warmup_max_steps", total_training_steps))
        computed = floor(total_training_steps * warmup_ratio)
        return max(warmup_min_steps, min(warmup_max_steps, computed))
    return int(training_cfg.get("warmup_steps", 0))


def _build_optimizer(model, config: Dict, logger):
    base_lr = float(config["training"]["learning_rate"])
    trajectory_lr = float(config["training"].get("trajectory_learning_rate", config["training"].get("proj_learning_rate", base_lr)))
    bridge_lr = float(config["training"].get("bridge_learning_rate", base_lr))
    weight_decay = float(config["training"]["weight_decay"])

    bridge_param_names = {
        "qformer_input_proj",
        "qformer_to_mlp1_proj",
    }
    trajectory_param_names = {
        "trajectory_backbone",
        "trajectory_token_projector",
    }

    bridge_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(pn in n for pn in bridge_param_names)
    ]
    trajectory_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(pn in n for pn in trajectory_param_names)
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and not any(pn in n for pn in bridge_param_names)
        and not any(pn in n for pn in trajectory_param_names)
    ]
    logger.info(
        "Pretrain param groups | trajectory=%s params @ lr=%s | bridge=%s params @ lr=%s | other=%s params @ lr=%s",
        sum(p.numel() for p in trajectory_params),
        trajectory_lr,
        sum(p.numel() for p in bridge_params),
        bridge_lr,
        sum(p.numel() for p in other_params),
        base_lr,
    )

    param_groups = []
    if trajectory_params:
        param_groups.append({"params": trajectory_params, "lr": trajectory_lr})
    if bridge_params:
        param_groups.append({"params": bridge_params, "lr": bridge_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})

    optimizer = AdamW(param_groups, weight_decay=weight_decay, foreach=False)
    enforce_safe_optimizer_param_groups(optimizer)
    return optimizer


def _save_pretrain_checkpoint(model, tokenizer, output_dir: str, optimizer=None, lr_scheduler=None):
    os.makedirs(output_dir, exist_ok=True)
    save_qformer_bridge(model, output_dir)
    save_trajectory_branch(model, output_dir)
    tokenizer.save_pretrained(output_dir)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))


def _write_training_state(output_dir: str, next_epoch: int, next_step: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump({"next_epoch": int(next_epoch), "next_step": int(next_step)}, f, indent=2, ensure_ascii=False)


def _write_early_stopping_state(output_dir: str, early_stopping: EarlyStoppingState):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, EARLY_STOPPING_STATE_FILENAME), "w", encoding="utf-8") as f:
        json.dump(
            {
                "patience": int(early_stopping.patience),
                "min_delta": float(early_stopping.min_delta),
                "best_val_loss": None if early_stopping.best_val_loss is None else float(early_stopping.best_val_loss),
                "best_epoch": None if early_stopping.best_epoch is None else int(early_stopping.best_epoch),
                "num_bad_epochs": int(early_stopping.num_bad_epochs),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def _load_early_stopping_state(checkpoint_dir: str) -> EarlyStoppingState | None:
    state_path = os.path.join(checkpoint_dir, EARLY_STOPPING_STATE_FILENAME)
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return EarlyStoppingState(
        patience=int(payload["patience"]),
        min_delta=float(payload["min_delta"]),
        best_val_loss=None if payload.get("best_val_loss") is None else float(payload["best_val_loss"]),
        best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
        num_bad_epochs=int(payload.get("num_bad_epochs", 0)),
    )


def _log_error_stats(logger, prefix: str, stats: Dict):
    logger.info(
        "%s sample errors | count=%s | rate=%.4f | requested=%s",
        prefix,
        stats["sample_error_count"],
        stats["sample_error_rate"],
        stats["requested_count"],
    )
    for example in stats["error_examples"]:
        logger.info("%s error example | %s", prefix, json.dumps(example, ensure_ascii=False))


def _print_debug_samples(logger, dataset, prefix: str, limit: int = 2):
    if len(dataset) == 0:
        return
    for idx in range(min(limit, len(dataset))):
        snapshot = dataset.get_debug_snapshot(idx)
        logger.info(
            "[DEBUG] %s sample %s | frame_path=%s | frame_id=%s",
            prefix,
            idx + 1,
            snapshot["frame_path"],
            snapshot["frame_id"],
        )
        logger.info("[DEBUG] question: %s", snapshot["question"])
        logger.info("[DEBUG] qformer_text: %s", snapshot["qformer_text"])
        logger.info("[DEBUG] answer: %s", snapshot["answer"])
        logger.info(
            "[DEBUG] trajectory | label_ids=%s | direction_ids=%s | object_mask=%s | numeric_feats=%s",
            snapshot["trajectory_label_ids"],
            snapshot["trajectory_direction_ids"],
            snapshot["trajectory_object_mask"],
            snapshot["trajectory_numeric_feats"],
        )


def eval_pretrain(model, val_loader, epoch: int, epochs: int, device: torch.device, logger):
    model.eval()
    total_eval_loss = 0.0
    total_eval_batches = 0
    val_loader.dataset.reset_error_stats()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Pretrain eval {epoch + 1}/{epochs}", leave=False):
            loss = forward_pretrain_batch(model, batch, device=device)
            total_eval_loss += float(loss.item())
            total_eval_batches += 1
    avg_eval_loss = total_eval_loss / total_eval_batches if total_eval_batches > 0 else float("nan")
    _log_error_stats(logger, "VAL", val_loader.dataset.consume_error_stats())
    model.train()
    if getattr(model, "qformer_enabled", False):
        model.qformer.eval()
        model.mlp1.eval()
    return avg_eval_loss


def train_pretrain(model, tokenizer, train_loader, val_loader, config: Dict, output_dir: str, logger):
    epochs = int(config["training"]["num_epochs"])
    accum_steps = int(config["training"]["gradient_accumulation_steps"])
    max_grad_norm = float(config["training"]["max_grad_norm"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = _build_optimizer(model, config, logger)
    total_training_steps = max((len(train_loader) * epochs) // max(accum_steps, 1), 1)
    warmup_steps = resolve_warmup_steps(config, total_training_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    logger.info("Trainable parameter summary:")
    for row in trainable_parameter_summary(model):
        logger.info(row)

    return optimizer, lr_scheduler


def validate_pretrain_resume_checkpoint(model, checkpoint_dir, logger):
    bridge_metadata = read_qformer_bridge_metadata(checkpoint_dir)
    traj_metadata = read_trajectory_branch_metadata(checkpoint_dir)
    if not os.path.exists(os.path.join(checkpoint_dir, "qformer_bridge.safetensors")):
        raise FileNotFoundError(f"Pretrain resume checkpoint is missing qformer_bridge.safetensors: {checkpoint_dir}")
    if not os.path.exists(os.path.join(checkpoint_dir, "trajectory_branch.safetensors")):
        raise FileNotFoundError(f"Pretrain resume checkpoint is missing trajectory_branch.safetensors: {checkpoint_dir}")

    stages = {m.get("stage") for m in (bridge_metadata, traj_metadata) if m}
    if stages and stages != {"pretrain"}:
        raise ValueError(f"Resume checkpoint stage metadata mismatch: {sorted(stages)}")
    if not stages:
        logger.warning("Pretrain resume checkpoint has no explicit stage metadata; treating it as a legacy compatible checkpoint.")

    checkpoint_mode = traj_metadata.get("fusion_mode")
    current_mode = getattr(model, "trajectory_fusion_mode", None)
    if checkpoint_mode and current_mode and checkpoint_mode != current_mode:
        raise ValueError(
            f"Pretrain resume fusion mode mismatch: checkpoint={checkpoint_mode}, current={current_mode}"
        )


def run_pretrain_training(model, tokenizer, train_loader, val_loader, config: Dict, output_dir: str, logger, resume_dir=None, start_epoch=0, start_step=0):
    epochs = int(config["training"]["num_epochs"])
    accum_steps = int(config["training"]["gradient_accumulation_steps"])
    max_grad_norm = float(config["training"]["max_grad_norm"])
    metrics_path = os.path.join(output_dir, "metrics.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = int(config["training"].get("early_stopping_patience", 0))
    min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    restore_best_checkpoint = bool(config["training"].get("restore_best_checkpoint", True))
    best_dir = os.path.join(output_dir, "best")
    last_dir = os.path.join(output_dir, "last")
    early_stopping = EarlyStoppingState(patience=patience, min_delta=min_delta) if patience > 0 else None
    metrics = {"train_loss": [], "val_loss": [], "epoch_summary": []}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    optimizer, lr_scheduler = train_pretrain(model, tokenizer, train_loader, val_loader, config, output_dir, logger)

    if resume_dir and os.path.exists(resume_dir):
        logger.info(f"Resuming pretrain from {resume_dir} | Epoch: {start_epoch+1}, Step: {start_step}")
        validate_pretrain_resume_checkpoint(model, resume_dir, logger)
        load_qformer_bridge(model, resume_dir, strict=True)
        align_qformer_bridge_runtime(model)
        load_trajectory_branch(model, resume_dir, strict=True)
        align_qformer_bridge_runtime(model)

        opt_path = os.path.join(resume_dir, "optimizer.pt")
        sch_path = os.path.join(resume_dir, "scheduler.pt")
        if os.path.exists(opt_path) and os.path.exists(sch_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            lr_scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
        else:
            logger.warning("No optimizer/scheduler states found in pretrain checkpoint. Starting with fresh optimizer states.")

        if early_stopping is not None:
            restored_early_stopping = _load_early_stopping_state(resume_dir)
            if restored_early_stopping is None:
                logger.warning(
                    "No %s found in pretrain checkpoint. Early stopping state will restart from scratch.",
                    EARLY_STOPPING_STATE_FILENAME,
                )
            else:
                early_stopping.best_val_loss = restored_early_stopping.best_val_loss
                early_stopping.best_epoch = restored_early_stopping.best_epoch
                early_stopping.num_bad_epochs = restored_early_stopping.num_bad_epochs
                logger.info(
                    "Restored early stopping state | best_epoch=%s | best_val_loss=%s | bad_epochs=%s/%s",
                    early_stopping.best_epoch,
                    f"{early_stopping.best_val_loss:.4f}" if early_stopping.best_val_loss is not None else "None",
                    early_stopping.num_bad_epochs,
                    early_stopping.patience,
                )

    for epoch in range(start_epoch, epochs):
        model.train()
        if getattr(model, "qformer_enabled", False):
            model.qformer.eval()
            model.mlp1.eval()
        optimizer.zero_grad()
        train_loader.dataset.reset_error_stats()
        accumulated_loss_for_log = 0.0
        global_steps_this_epoch = 0

        batch_iterator = iter(train_loader)
        if epoch == start_epoch and start_step > 0:
            logger.info(f"Skipping {start_step} batches to resume pretrain state...")
            for _ in tqdm(range(start_step), desc="Skipping to pretrain resume point", leave=False):
                next(batch_iterator)
            initial_step = start_step
        else:
            initial_step = 0

        progress_bar = tqdm(batch_iterator, desc=f"Pretrain epoch {epoch + 1}/{epochs}", total=len(train_loader), initial=initial_step)
        for step, batch in enumerate(progress_bar, start=initial_step + 1):
            loss = forward_pretrain_batch(model, batch, device=device)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            (loss / accum_steps).backward()
            accumulated_loss_for_log += float(loss.item())

            if step % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_steps_this_epoch += 1
                avg_loss = accumulated_loss_for_log / accum_steps
                metrics["train_loss"].append({"epoch": epoch + 1, "step": global_steps_this_epoch, "loss": round(avg_loss, 6)})
                accumulated_loss_for_log = 0.0

        train_stats = train_loader.dataset.consume_error_stats()
        _log_error_stats(logger, "TRAIN", train_stats)

        val_loss = eval_pretrain(model, val_loader, epoch, epochs, device, logger)
        metrics["val_loss"].append({"epoch": epoch + 1, "loss": round(val_loss, 6)})

        epoch_losses = [item["loss"] for item in metrics["train_loss"] if item["epoch"] == epoch + 1]
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
        metrics["epoch_summary"].append(
            {"epoch": epoch + 1, "avg_train_loss": round(avg_epoch_loss, 6), "val_loss": round(val_loss, 6)}
        )
        logger.info(
            "Pretrain epoch %s summary | avg_train_loss=%.4f | val_loss=%.4f",
            epoch + 1,
            avg_epoch_loss,
            val_loss,
        )
        improved = False
        should_stop = False
        if early_stopping is not None:
            improved, should_stop = early_stopping.update(epoch=epoch + 1, val_loss=val_loss)
            logger.info(
                "Early stopping status | improved=%s | best_epoch=%s | best_val_loss=%.4f | bad_epochs=%s/%s",
                improved,
                early_stopping.best_epoch,
                early_stopping.best_val_loss if early_stopping.best_val_loss is not None else float("nan"),
                early_stopping.num_bad_epochs,
                early_stopping.patience,
            )

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        if os.path.exists(last_dir):
            shutil.rmtree(last_dir)
        _save_pretrain_checkpoint(model, tokenizer, last_dir, optimizer=optimizer, lr_scheduler=lr_scheduler)
        _write_training_state(last_dir, next_epoch=epoch + 1, next_step=0)
        if early_stopping is not None:
            _write_early_stopping_state(last_dir, early_stopping)
        logger.info("Saved latest pretrain checkpoint to %s", last_dir)
        if improved:
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            _save_pretrain_checkpoint(model, tokenizer, best_dir, optimizer=optimizer, lr_scheduler=lr_scheduler)
            _write_training_state(best_dir, next_epoch=epoch + 1, next_step=0)
            if early_stopping is not None:
                _write_early_stopping_state(best_dir, early_stopping)
            logger.info("Updated best pretrain checkpoint at %s", best_dir)
        if should_stop:
            logger.info("Early stopping triggered at epoch %s", epoch + 1)
            break

    if restore_best_checkpoint and os.path.exists(best_dir):
        logger.info("Restoring best checkpoint from %s", best_dir)
        load_qformer_bridge(model, best_dir, strict=True)
        align_qformer_bridge_runtime(model)
        load_trajectory_branch(model, best_dir, strict=True)
        align_qformer_bridge_runtime(model)


def main():
    set_seed(42)
    args = parse_args()
    config = load_config(args.config)
    output_dir = build_output_dir(config)
    init_logger(output_dir)
    logger = get_logger()
    resume_dir, start_epoch, start_step = resolve_resume_config(args, logger)

    model, tokenizer = build_model_and_tokenizer(config, logger)
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(config, model, tokenizer)
    _print_debug_samples(logger, train_dataset, "train")
    _print_debug_samples(logger, val_dataset, "val")

    logger.info(
        "Pretrain runtime check | qformer_enabled=%s | trajectory_enabled=%s | trajectory_mode=%s | num_image_token=%s | train_rows=%s | val_rows=%s",
        getattr(model, "qformer_enabled", False),
        getattr(model, "trajectory_enabled", False),
        getattr(model, "trajectory_fusion_mode", "disabled"),
        getattr(model, "num_image_token", "unknown"),
        len(train_dataset),
        len(val_dataset),
    )

    run_pretrain_training(
        model,
        tokenizer,
        train_loader,
        val_loader,
        config,
        output_dir,
        logger,
        resume_dir=resume_dir,
        start_epoch=start_epoch,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()
