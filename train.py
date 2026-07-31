import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import datetime
import re
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from logutil import get_logger, init_logger
from optimizer_state_utils import (
    count_optimizer_state_tensors_on_cpu,
    enforce_safe_optimizer_param_groups,
    export_sanitized_optimizer_state_dict,
    move_optimizer_state_to_param_device,
    sanitize_optimizer_state_dict,
)
from pretrain_checkpoint_verify import verify_loaded_pretrain_modules


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

logger = None

from wad_dataset import build_dataset
from model.conversation import get_conv_template
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train InternVL with optional checkpoint resume.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint dir to resume from. Omit this flag to train from the default LoRA/base setup.",
    )
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default=None,
        help="Optional pretrain checkpoint dir or HF repo id. Loads only qformer_bridge + trajectory_branch, then initializes fresh LoRA.",
    )
    parser.add_argument("--start_epoch", type=int, default=None, help="Zero-based epoch index to resume from.")
    parser.add_argument("--start_step", type=int, default=None, help="Batch step inside the resume epoch.")
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_runtime(config):
    base_out_dir = config["training"]["output_dir"]
    output_dir = f'{base_out_dir}/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
    os.makedirs(output_dir, exist_ok=True)
    init_logger(output_dir)
    return output_dir, get_logger()


def infer_resume_position(checkpoint_dir):
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


def resolve_checkpoint_path(checkpoint):
    if not checkpoint:
        return None
    if os.path.exists(checkpoint):
        logger.info(f"Using local checkpoint path: {checkpoint}")
        return checkpoint
    logger.info(f"Checkpoint is not a local path. Downloading from Hugging Face: {checkpoint}")
    return snapshot_download(
        repo_id=checkpoint,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "qformer_bridge.safetensors",
            "qformer_bridge_config.json",
            "trajectory_branch.safetensors",
            "trajectory_branch_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )


def resolve_resume_config(args, config):
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    start_epoch = args.start_epoch
    start_step = args.start_step

    inferred_epoch, inferred_step = infer_resume_position(checkpoint) if checkpoint else (None, None)
    if start_epoch is None:
        start_epoch = inferred_epoch if inferred_epoch is not None else 0
    if start_step is None:
        start_step = inferred_step if inferred_step is not None else 0

    return checkpoint, int(start_epoch or 0), int(start_step or 0)


def build_fresh_lora_model(language_model, config, logger):
    lora_cfg = config["model"]["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    logger.info("Initializing a fresh LoRA adapter from config.")
    return get_peft_model(language_model, peft_config)


def validate_pretrain_checkpoint_for_finetune(model, checkpoint_dir, logger):
    bridge_metadata = read_qformer_bridge_metadata(checkpoint_dir)
    traj_metadata = read_trajectory_branch_metadata(checkpoint_dir)

    if not os.path.exists(os.path.join(checkpoint_dir, "qformer_bridge.safetensors")):
        raise FileNotFoundError(f"Pretrain checkpoint is missing qformer_bridge.safetensors: {checkpoint_dir}")
    if not os.path.exists(os.path.join(checkpoint_dir, "trajectory_branch.safetensors")):
        raise FileNotFoundError(f"Pretrain checkpoint is missing trajectory_branch.safetensors: {checkpoint_dir}")

    stages = {m.get("stage") for m in (bridge_metadata, traj_metadata) if m}
    if stages:
        if stages != {"pretrain"}:
            raise ValueError(f"Pretrain checkpoint stage metadata mismatch: {sorted(stages)}")
    else:
        logger.warning("Pretrain checkpoint has no explicit stage metadata; treating it as a legacy pretrain-compatible checkpoint.")

    checkpoint_mode = traj_metadata.get("fusion_mode")
    current_mode = getattr(model, "trajectory_fusion_mode", None)
    if checkpoint_mode and current_mode and checkpoint_mode != current_mode:
        raise ValueError(
            f"Pretrain checkpoint fusion mode mismatch: checkpoint={checkpoint_mode}, current={current_mode}"
        )

    return bridge_metadata, traj_metadata


def log_pretrain_checkpoint_verification(model, checkpoint_dir, logger):
    verification = verify_loaded_pretrain_modules(model, checkpoint_dir)
    for section_name in ("bridge", "trajectory"):
        section = verification[section_name]
        logger.info(
            "Pretrain %s verification | all_matched=%s | checked=%s",
            section_name,
            section["all_matched"],
            len(section["checked_keys"]),
        )
        for item in section["checked_keys"]:
            logger.info(
                "Pretrain %s key check | key=%s | matched=%s | model_norm=%.6f | checkpoint_norm=%.6f",
                section_name,
                item["key"],
                item["matched"],
                item["model_norm"],
                item["checkpoint_norm"],
            )
    return verification


def compute_sequence_loss(logits, labels, loss_mode: str, label_smoothing: float):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous().to(shifted_logits.device)
    flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flat_labels = shifted_labels.view(-1)
    if loss_mode == "label_smoothing":
        return F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
    if loss_mode == "cross_entropy":
        return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
    raise ValueError(f"Unsupported loss_mode: {loss_mode}")


def build_optimizer_param_groups(model, *, lora_lr: float, bridge_lr: float, trajectory_lr: float):
    groups = {
        "trajectory": {"params": [], "param_names": [], "lr": trajectory_lr, "name": "trajectory"},
        "bridge": {"params": [], "param_names": [], "lr": bridge_lr, "name": "bridge"},
        "lora_rest": {"params": [], "param_names": [], "lr": lora_lr, "name": "lora_rest"},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(part in name for part in ("trajectory_backbone", "trajectory_token_projector")):
            group = groups["trajectory"]
        elif any(part in name for part in ("qformer_input_proj", "qformer_to_mlp1_proj")):
            group = groups["bridge"]
        else:
            group = groups["lora_rest"]
        group["params"].append(param)
        group["param_names"].append(name)

    grouped_ids = [id(param) for group in groups.values() for param in group["params"]]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("Duplicate trainable parameter detected across optimizer groups.")
    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}
    if set(grouped_ids) != trainable_ids:
        raise ValueError("Optimizer groups do not cover exactly all trainable parameters.")
    return [group for group in groups.values() if group["params"]]


def maybe_pad(inner_lists, padding_value):
    tensor_list = [torch.tensor(inner_list, dtype=torch.long) for inner_list in inner_lists]
    return pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)


class CollaterFn:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.log_token_stats = False
        self.token_log_remaining = 0
        self.alter_only = False

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
                logger.info(
                    "[INFO][ALTER_ONLY=%s] Image token stats | frames=%s | tiles_per_frame=%s | query_tokens_per_tile=%s | total_image_tokens=%s",
                    self.alter_only,
                    len(pixel_values),
                    num_patches_list,
                    self.model.num_image_token,
                    total_image_tokens_in_sample,
                )
                logger.info(
                    "[INFO][ALTER_ONLY=%s] Text tokens - input: %s, answer: %s, total: %s",
                    self.alter_only,
                    len(input_ids),
                    len(answer_ids),
                    total_sequence_length,
                )
                if self.token_log_remaining > 0:
                    self.token_log_remaining -= 1

            label_ids = [-100] * len(input_ids) + answer_ids + [eos_token_id]
            input_ids = input_ids + answer_ids + [eos_token_id]
            attention_mask = [1] * len(input_ids)
            assert len(input_ids) == len(attention_mask) == len(label_ids)

            label_ids_batch.append(label_ids)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pixel_values_batch.append(torch.cat(pixel_values, dim=0))
            if getattr(self.model, "qformer_enabled", False):
                qformer_text = sample.get("qformer_text", question.replace("<image>", "").strip())
                qformer_texts.extend([qformer_text] * total_tiles)
            if getattr(self.model, "trajectory_enabled", False):
                label_ids = sample.get("trajectory_label_ids")
                direction_ids = sample.get("trajectory_direction_ids")
                numeric_feats = sample.get("trajectory_numeric_feats")
                object_mask = sample.get("trajectory_object_mask")
                if label_ids is None or direction_ids is None or numeric_feats is None or object_mask is None:
                    raise ValueError("Trajectory-enabled sample is missing trajectory fields.")
                for _ in range(total_tiles):
                    trajectory_label_ids_batch.append(torch.as_tensor(label_ids, dtype=torch.long))
                    trajectory_direction_ids_batch.append(torch.as_tensor(direction_ids, dtype=torch.long))
                    trajectory_numeric_feats_batch.append(torch.as_tensor(numeric_feats, dtype=torch.float32))
                    trajectory_object_mask_batch.append(torch.as_tensor(object_mask, dtype=torch.long))

        input_ids_tensor = maybe_pad(input_ids_batch, eot_token_id)
        label_ids_tensor = maybe_pad(label_ids_batch, -100)
        attention_mask_tensor = maybe_pad(attention_mask_batch, 0)
        pixel_values_tensor = torch.cat(pixel_values_batch)
        qformer_inputs = None
        trajectory_inputs = None
        if getattr(self.model, "qformer_enabled", False):
            qformer_inputs = self.model.encode_qformer_texts(qformer_texts)
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


def test_model(model, tokenizer, val_loader_with_shuffle, shuffle=False):
    model.eval()
    with torch.no_grad():
        total_test_batches = 0
        for batch in tqdm(val_loader_with_shuffle):
            _, _, _, _, _, _, samples = batch
            for sample in samples:
                pixel_values = torch.cat(sample["pixel_values"], dim=0).to(torch.bfloat16).cuda()
                generation_config = dict(max_new_tokens=512, do_sample=False)
                question = f"{sample['question']}"
                if getattr(model, "qformer_enabled", False):
                    q_ids, q_mask = model.encode_qformer_texts(
                        [sample.get("qformer_text", question.replace("<image>", "").strip())] * pixel_values.shape[0],
                        device=pixel_values.device,
                    )
                    model.set_qformer_text(q_ids, q_mask)
                if getattr(model, "trajectory_enabled", False):
                    model.set_trajectory_inputs(
                        sample["trajectory_label_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                        sample["trajectory_direction_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                        sample["trajectory_numeric_feats"].unsqueeze(0).repeat(pixel_values.shape[0], 1, 1).cuda(),
                        sample["trajectory_object_mask"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                    )
                response = model.chat(tokenizer, pixel_values, question, generation_config)
                if getattr(model, "qformer_enabled", False):
                    model.clear_qformer_text()
                if getattr(model, "trajectory_enabled", False):
                    model.clear_trajectory_inputs()
                question_token_count = len(tokenizer.encode(question, add_special_tokens=False))
                response_token_count = len(tokenizer.encode(response, add_special_tokens=False))
                ground_truth_token_count = len(tokenizer.encode(sample["answer"], add_special_tokens=False))
                logger.info(
                    f"\nToken stats | question: {question_token_count} | "
                    f"response: {response_token_count} | ground_truth: {ground_truth_token_count}"
                )
                logger.info(f'\nUser: {question}\nAssistant: {response}\nGround truth:{sample["answer"]}\n\n')
            total_test_batches += 1
            if total_test_batches == 2:
                break


def eval_model(model, val_loader, step, epoch, epochs, loss_mode: str, label_smoothing: float):
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0
        total_eval_batchs = 0
        eval_desc = f"Eval @ step {step} | epoch {epoch + 1}/{epochs}"
        for batch in tqdm(val_loader, desc=eval_desc, leave=False):
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, qformer_inputs, trajectory_inputs, _ = batch
            input_ids_batch = input_ids_batch.cuda()
            label_ids_batch = label_ids_batch.cuda()
            attention_mask_batch = attention_mask_batch.cuda()
            pixel_values_batch = pixel_values_batch.to(torch.bfloat16).cuda()
            image_flags_batch = torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long).cuda()
            if getattr(model, "qformer_enabled", False) and qformer_inputs is not None:
                model.set_qformer_text(qformer_inputs[0].cuda(), qformer_inputs[1].cuda())
            if getattr(model, "trajectory_enabled", False) and trajectory_inputs is not None:
                model.set_trajectory_inputs(
                    trajectory_inputs[0].cuda(),
                    trajectory_inputs[1].cuda(),
                    trajectory_inputs[2].cuda(),
                    trajectory_inputs[3].cuda(),
                )

            outputs = model(
                input_ids=input_ids_batch,
                pixel_values=pixel_values_batch,
                labels=label_ids_batch,
                image_flags=image_flags_batch,
                return_dict=True,
            )
            if getattr(model, "qformer_enabled", False):
                model.clear_qformer_text()
            if getattr(model, "trajectory_enabled", False):
                model.clear_trajectory_inputs()
            loss = compute_sequence_loss(
                logits=outputs.logits,
                labels=label_ids_batch,
                loss_mode=loss_mode,
                label_smoothing=label_smoothing,
            )
            total_eval_loss += loss.item()
            total_eval_batchs += 1
            if total_eval_batchs == 200:
                break
        avg_eval_loss = total_eval_loss / total_eval_batchs if total_eval_batchs > 0 else float("nan")
        logger.info(f"Validation loss after {step} batches training in epoch {epoch + 1}/{epochs}: {avg_eval_loss:.4f}")
    model.train()
    if getattr(model, "qformer_enabled", False):
        model.qformer.eval()
        model.mlp1.eval()
    return avg_eval_loss


def train_model(model, tokenizer, train_loader, val_loader, val_loader_with_shuffle, config, output_dir, resume_dir=None, start_epoch=0, start_step=0):
    epochs = config["training"]["num_epochs"]
    lr = float(config["training"]["learning_rate"])
    accum_steps = config["training"]["gradient_accumulation_steps"]
    weight_decay = float(config["training"]["weight_decay"])
    warmup_steps = config["training"]["warmup_steps"]
    max_grad_norm = float(config["training"]["max_grad_norm"])
    eval_steps = config["training"].get("eval_steps")
    log_token_stats = bool(config["training"].get("log_token_stats", False))
    loss_mode = str(config["training"].get("loss_mode", "cross_entropy"))
    label_smoothing = float(config["training"].get("label_smoothing", 0.0))
    lora_lr = float(config["training"].get("lora_learning_rate", lr))
    bridge_lr = float(config["training"].get("bridge_learning_rate", config["training"].get("proj_learning_rate", lr)))
    trajectory_lr = float(config["training"].get("trajectory_learning_rate", config["training"].get("proj_learning_rate", lr)))
    metrics_path = os.path.join(output_dir, "metrics.json")

    def save_metrics(metrics: dict):
        import json
        with open(metrics_path, "w", encoding="utf-8") as _f:
            json.dump(metrics, _f, indent=2, ensure_ascii=False)

    metrics = {"train_loss": [], "val_loss": [], "epoch_summary": []}

    logger.info(f"Total params: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(
        "Training config: loss_mode=%s, label_smoothing=%.3f, accum_steps=%s, weight_decay=%s, "
        "lora_lr=%s, bridge_lr=%s, trajectory_lr=%s",
        loss_mode,
        label_smoothing,
        accum_steps,
        weight_decay,
        lora_lr,
        bridge_lr,
        trajectory_lr,
    )

    optimizer_groups = build_optimizer_param_groups(
        model,
        lora_lr=lora_lr,
        bridge_lr=bridge_lr,
        trajectory_lr=trajectory_lr,
    )
    logger.info(
        "Param groups | %s",
        " | ".join(
            f"{group['name']}: {sum(p.numel() for p in group['params']):,} params @ lr={group['lr']}"
            for group in optimizer_groups
        ),
    )

    optimizer = AdamW(
        optimizer_groups,
        weight_decay=weight_decay,
        foreach=False,
    )
    enforce_safe_optimizer_param_groups(optimizer)
    save_steps = config["training"].get("save_steps")

    total_training_steps = (len(train_loader) * epochs) // accum_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    if resume_dir and os.path.exists(resume_dir):
        logger.info(f"Resuming training from {resume_dir} | Epoch: {start_epoch+1}, Step: {start_step}")
        if getattr(model, "qformer_enabled", False):
            load_qformer_bridge(model, resume_dir, strict=True)
            align_qformer_bridge_runtime(model)
            logger.info("Loaded Q-Former bridge states successfully!")
        if getattr(model, "trajectory_enabled", False):
            load_trajectory_branch(model, resume_dir, strict=True)
            align_qformer_bridge_runtime(model)
            logger.info("Loaded trajectory branch states successfully!")

        opt_path = os.path.join(resume_dir, "optimizer.pt")
        sch_path = os.path.join(resume_dir, "scheduler.pt")

        if os.path.exists(opt_path) and os.path.exists(sch_path):
            optimizer_state_dict, converted, overridden_groups = sanitize_optimizer_state_dict(
                torch.load(opt_path, map_location="cpu")
            )
            optimizer.load_state_dict(optimizer_state_dict)
            overridden_groups_after_load = enforce_safe_optimizer_param_groups(optimizer)
            moved = move_optimizer_state_to_param_device(optimizer)
            lr_scheduler.load_state_dict(torch.load(sch_path))
            if converted:
                logger.info("Sanitized %s optimizer state tensors to float32 after load.", converted)
            if overridden_groups:
                logger.info("Overrode foreach/fused flags in %s optimizer param_groups before load.", overridden_groups)
            if overridden_groups_after_load:
                logger.info("Re-applied safe foreach/fused flags to %s optimizer param_groups after load.", overridden_groups_after_load)
            if moved:
                logger.info("Moved %s optimizer state tensors to parameter devices after load.", moved)
            remaining_cpu_tensors = count_optimizer_state_tensors_on_cpu(optimizer)
            logger.info("Optimizer state CPU tensor count after load: %s", remaining_cpu_tensors)
            logger.info("Loaded Optimizer and Scheduler states successfully!")
        else:
            logger.warning("No Optimizer/Scheduler states found in checkpoint. Starting with fresh states.")

    for epoch in range(start_epoch, epochs):
        model.train()
        if getattr(model, "qformer_enabled", False):
            model.qformer.eval()
            model.mlp1.eval()
        optimizer.zero_grad()

        accumulated_loss_for_log = 0.0
        set_seed(42)
        logger.info("Epoch seed fixed | epoch=%s | seed=42", epoch + 1)
        batch_iterator = iter(train_loader)
        train_loader.collate_fn.log_token_stats = False

        if epoch == start_epoch and start_step > 0:
            logger.info(f" Skipping {start_step} batches to resume state...")
            for _ in tqdm(range(start_step), desc="Skipping to resume point", leave=False):
                next(batch_iterator)
            i = start_step
        else:
            i = 0
            
        train_loader.collate_fn.log_token_stats = log_token_stats
        progress_bar = tqdm(batch_iterator, desc=f"Training Epoch {epoch + 1}/{epochs}", total=len(train_loader), initial=i)
        for batch in progress_bar:
            i += 1
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, qformer_inputs, trajectory_inputs, _ = batch

            input_ids_batch = input_ids_batch.cuda()
            label_ids_batch = label_ids_batch.cuda()
            attention_mask_batch = attention_mask_batch.cuda()
            pixel_values_batch = pixel_values_batch.to(torch.bfloat16).cuda()
            image_flags_batch = torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long).cuda()
            if getattr(model, "qformer_enabled", False) and qformer_inputs is not None:
                model.set_qformer_text(qformer_inputs[0].cuda(), qformer_inputs[1].cuda())
            if getattr(model, "trajectory_enabled", False) and trajectory_inputs is not None:
                model.set_trajectory_inputs(
                    trajectory_inputs[0].cuda(),
                    trajectory_inputs[1].cuda(),
                    trajectory_inputs[2].cuda(),
                    trajectory_inputs[3].cuda(),
                )

            outputs = model(
                input_ids=input_ids_batch,
                pixel_values=pixel_values_batch,
                labels=label_ids_batch,
                image_flags=image_flags_batch,
                return_dict=True,
            )
            if getattr(model, "qformer_enabled", False):
                model.clear_qformer_text()
            if getattr(model, "trajectory_enabled", False):
                model.clear_trajectory_inputs()

            raw_loss = compute_sequence_loss(
                logits=outputs.logits,
                labels=label_ids_batch,
                loss_mode=loss_mode,
                label_smoothing=label_smoothing,
            )
            loss = raw_loss / accum_steps
            progress_bar.set_postfix(loss=f"{raw_loss.item():.4f}")
            loss.backward()

            accumulated_loss_for_log += raw_loss.item()

            if i % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                avg_loss = accumulated_loss_for_log / accum_steps
                global_step = i // accum_steps
                metrics["train_loss"].append({"step": global_step, "epoch": epoch + 1, "loss": round(avg_loss, 6)})
                if global_step % 50 == 0:
                    logger.info(f"Step {global_step} | Avg Loss: {avg_loss:.4f}")
                    save_metrics(metrics)
                accumulated_loss_for_log = 0.0

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if eval_steps and i % eval_steps == 0:
                logger.info(f"Running evaluation at step {i}...")
                val_loss = eval_model(model, val_loader, i, epoch, epochs, loss_mode, label_smoothing)
                if val_loss is not None:
                    metrics["val_loss"].append({"step": i, "epoch": epoch + 1, "loss": round(val_loss, 6)})
                    save_metrics(metrics)

            if save_steps and i % save_steps == 0:
                step_save_dir = f"{output_dir}/epoch_{epoch+1}_step_{i}/"
                os.makedirs(step_save_dir, exist_ok=True)
                logger.info(f"Saving model, tokenizer, opt, scheduler at step {i} to {step_save_dir}")

                model.language_model.save_pretrained(step_save_dir)
                save_qformer_bridge(model, step_save_dir)
                save_trajectory_branch(model, step_save_dir)
                tokenizer.save_pretrained(step_save_dir)
                optimizer_state_dict, converted, overridden_groups = export_sanitized_optimizer_state_dict(optimizer)
                if converted:
                    logger.info("Sanitized %s optimizer state tensors to float32 before save.", converted)
                if overridden_groups:
                    logger.info("Normalized foreach/fused flags in %s optimizer param_groups before save.", overridden_groups)
                torch.save(optimizer_state_dict, os.path.join(step_save_dir, "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), os.path.join(step_save_dir, "scheduler.pt"))

        epoch_save_dir = f"{output_dir}/epoch_{epoch+1}/"
        os.makedirs(epoch_save_dir, exist_ok=True)
        logger.info(f"Saving model and tokenizer for epoch {epoch+1} to {epoch_save_dir}")
        model.language_model.save_pretrained(epoch_save_dir)
        save_qformer_bridge(model, epoch_save_dir)
        save_trajectory_branch(model, epoch_save_dir)
        tokenizer.save_pretrained(epoch_save_dir)
        optimizer_state_dict, converted, overridden_groups = export_sanitized_optimizer_state_dict(optimizer)
        if converted:
            logger.info("Sanitized %s optimizer state tensors to float32 before save.", converted)
        if overridden_groups:
            logger.info("Normalized foreach/fused flags in %s optimizer param_groups before save.", overridden_groups)
        torch.save(optimizer_state_dict, os.path.join(epoch_save_dir, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(epoch_save_dir, "scheduler.pt"))

        epoch_train = [e["loss"] for e in metrics["train_loss"] if e["epoch"] == epoch + 1]
        avg_epoch_loss = sum(epoch_train) / len(epoch_train) if epoch_train else float("nan")
        metrics["epoch_summary"].append({"epoch": epoch + 1, "avg_train_loss": round(avg_epoch_loss, 6)})
        logger.info(f"Epoch {epoch+1} summary | avg_train_loss={avg_epoch_loss:.4f}")
        save_metrics(metrics)


if __name__ == "__main__":
    global logger
    args = parse_args()
    config = load_config(args.config)
    output_dir, logger = initialize_runtime(config)
    if args.checkpoint and args.pretrain_checkpoint:
        raise ValueError("Use either --checkpoint for finetune resume or --pretrain_checkpoint for pretrain preload, not both.")
    resume_dir, start_epoch, start_step = resolve_resume_config(args, config)
    pretrain_checkpoint_dir = resolve_checkpoint_path(args.pretrain_checkpoint) if args.pretrain_checkpoint else None
    model_name_or_path = config["model"]["name"]
    batch_size = config["training"]["batch_size"]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["quantization"]["enabled"],
        bnb_4bit_compute_dtype=torch.bfloat16 if config["model"]["quantization"]["compute_dtype"] == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=config["model"]["quantization"]["double_quant"],
        bnb_4bit_quant_type=config["model"]["quantization"]["type"],
    )

    logger.info(f"Loading model {model_name_or_path} in 4-bit...")
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
    if qformer_enabled(config):
        attach_qformer_bridge(model, config, logger=logger)
        align_qformer_bridge_runtime(model)

    logger.info("Applying LoRA...")
    model.language_model = prepare_model_for_kbit_training(model.language_model)

    if hasattr(model.language_model, "get_input_embeddings"):
        model.language_model.get_input_embeddings().to(torch.bfloat16)

    if resume_dir:
        logger.info(f"Loading LoRA adapter from checkpoint: {resume_dir}")
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            resume_dir,
            is_trainable=True,
        )
    else:
        model.language_model = build_fresh_lora_model(model.language_model, config, logger)
        if pretrain_checkpoint_dir:
            logger.info(f"Preloading qformer bridge + trajectory branch from pretrain checkpoint: {pretrain_checkpoint_dir}")
            validate_pretrain_checkpoint_for_finetune(model, pretrain_checkpoint_dir, logger)
            load_qformer_bridge(model, pretrain_checkpoint_dir, strict=True)
            align_qformer_bridge_runtime(model)
            if getattr(model, "trajectory_enabled", False):
                load_trajectory_branch(model, pretrain_checkpoint_dir, strict=True)
                align_qformer_bridge_runtime(model)
            log_pretrain_checkpoint_verification(model, pretrain_checkpoint_dir, logger)

    apply_mode_gated_trajectory_trainability(model, logger)

    model.language_model.print_trainable_parameters()
    model.train()

    logger.info("Building dataset...")
    train_dataset, val_dataset = build_dataset(config)

    collate_fn_wrapper = CollaterFn(tokenizer, model)
    collate_fn_wrapper.log_token_stats = bool(config["training"].get("log_token_stats", False))
    collate_fn_wrapper.token_log_remaining = int(config["training"].get("token_log_batches", 0))
    collate_fn_wrapper.alter_only = bool(config["data"].get("alter_only", False))

    logger.info(
        "Runtime check | qformer_enabled=%s | trajectory_enabled=%s | trajectory_mode=%s | trajectory_source=%s | num_image_token=%s | qformer_tokens=%s | log_token_stats=%s | token_log_batches=%s | alter_only=%s | loss_mode=%s | label_smoothing=%.3f",
        getattr(model, "qformer_enabled", False),
        getattr(model, "trajectory_enabled", False),
        getattr(model, "trajectory_fusion_mode", "disabled"),
        getattr(model, "trajectory_source_file", "n/a"),
        getattr(model, "num_image_token", "unknown"),
        getattr(model, "qformer_num_query_tokens", getattr(model, "num_image_token", "unknown")),
        collate_fn_wrapper.log_token_stats,
        collate_fn_wrapper.token_log_remaining,
        collate_fn_wrapper.alter_only,
        config["training"].get("loss_mode", "cross_entropy"),
        float(config["training"].get("label_smoothing", 0.0)),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn_wrapper, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn_wrapper, shuffle=False
    )

    val_loader_with_shuffle = DataLoader(
        val_dataset, batch_size=1, collate_fn=collate_fn_wrapper, shuffle=True
    )

    logger.info("STARTING TRAINING...")
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        val_loader_with_shuffle=val_loader_with_shuffle,
        config=config,
        output_dir=output_dir,
        resume_dir=resume_dir,
        start_epoch=start_epoch,
        start_step=start_step,
    )
