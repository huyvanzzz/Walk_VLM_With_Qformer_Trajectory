import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file
from torch import nn


TRAJECTORY_BRANCH_WEIGHTS_NAME = "trajectory_branch.safetensors"
TRAJECTORY_BRANCH_CONFIG_NAME = "trajectory_branch_config.json"
TRAJECTORY_PAD_UNK_LABEL = "<PAD_UNK>"
TRAJECTORY_PAD_UNK_ID = 0
TRAJECTORY_NUM_OBJECTS = 6
TRAJECTORY_NUMERIC_DIM = 6
TRAJECTORY_CANONICAL_FILENAME = "results_botsort_top6_sorted.json"
TRAJECTORY_CANONICAL_JSONL_FILENAME = "results_botsort_top6_sorted.jsonl"
TRAJECTORY_NUMERIC_FIELDS = (
    "x1",
    "y1",
    "x2",
    "y2",
    "movement_angle",
    "speed_percent",
)


def trajectory_enabled(config: Dict) -> bool:
    traj_cfg = dict(config.get("trajectory", config.get("model", {}).get("trajectory", {})))
    return bool(traj_cfg.get("enabled", False))


def get_trajectory_config(config: Dict) -> Dict:
    traj_cfg = dict(config.get("trajectory", config.get("model", {}).get("trajectory", {})))
    traj_cfg.setdefault("enabled", False)
    traj_cfg.setdefault("fusion_mode", "concat")
    traj_cfg.setdefault("source_file", "image")
    traj_cfg.setdefault("num_objects", TRAJECTORY_NUM_OBJECTS)
    traj_cfg.setdefault("numeric_dim", TRAJECTORY_NUMERIC_DIM)
    traj_cfg.setdefault("d_cat", 32)
    traj_cfg.setdefault("d_dir", 16)
    traj_cfg.setdefault("d_numeric_hidden", 64)
    traj_cfg.setdefault("d_traj", 128)
    traj_cfg.setdefault("num_heads", 4)
    traj_cfg.setdefault("num_layers", 2)
    traj_cfg.setdefault("ffn_dim", 256)
    traj_cfg.setdefault("dropout", 0.0)
    if traj_cfg["enabled"] and str(traj_cfg["fusion_mode"]) != "concat":
        raise ValueError(
            f"Unsupported trajectory fusion_mode for concat-only public repo: {traj_cfg['fusion_mode']}"
        )
    return traj_cfg


def build_trajectory_source_from_config(config: Dict):
    if not trajectory_enabled(config):
        return None
    traj_cfg = get_trajectory_config(config)
    return TrajectorySource.from_file(str(traj_cfg["source_file"]))


def resolve_trajectory_source_path(source_file: str) -> str:
    normalized = os.path.normpath(str(source_file))
    if os.path.isdir(normalized):
        json_path = os.path.join(normalized, TRAJECTORY_CANONICAL_FILENAME)
        if os.path.exists(json_path):
            return json_path
        return os.path.join(normalized, TRAJECTORY_CANONICAL_JSONL_FILENAME)
    return normalized


def _normalize_flat_jsonl_records(records: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, int], List[Dict]] = {}
    for idx, record in enumerate(records):
        for field in ("folder_id", "frame_id", "label", "boxs", "relative_position", "movement_angle", "speed_percent"):
            if field not in record:
                raise ValueError(f"Flat trajectory jsonl row #{idx} is missing required field: {field}")
        key = (str(record["folder_id"]), int(record["frame_id"]))
        grouped.setdefault(key, []).append(record)

    normalized_records = []
    for (folder_id, frame_id), rows in grouped.items():
        if len(rows) > TRAJECTORY_NUM_OBJECTS:
            raise ValueError(
                f"Flat trajectory jsonl has {len(rows)} rows for {(folder_id, frame_id)}; max is {TRAJECTORY_NUM_OBJECTS}."
            )
        objects = []
        for row in rows:
            objects.append(
                _normalize_object_payload(row, key=(folder_id, frame_id), movement_is_degrees=True)
            )
        normalized_records.append(
            {
                "folder_id": folder_id,
                "frame_id": frame_id,
                "objects": objects,
            }
        )
    return normalized_records


def _normalize_object_payload(obj: Dict, key: Optional[Tuple[str, int]] = None, movement_is_degrees: bool = False) -> Dict:
    bbox = obj.get("boxs")
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Invalid 'boxs' value for {key or 'trajectory object'}: {bbox}")
        x1, y1, x2, y2 = bbox
    else:
        missing = [field for field in ("x1", "y1", "x2", "y2") if field not in obj]
        if missing:
            raise ValueError(
                f"Trajectory object in {key or 'record'} is missing bbox fields: {missing}"
            )
        x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]

    movement_angle = float(obj["movement_angle"])
    if movement_is_degrees:
        movement_angle = movement_angle / 180.0

    return {
        "label": str(obj["label"]),
        "relative_position": str(obj["relative_position"]),
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "movement_angle": movement_angle,
        "speed_percent": float(obj["speed_percent"]),
    }


def _empty_encoded_sample(
    num_objects: int = TRAJECTORY_NUM_OBJECTS,
    numeric_dim: int = TRAJECTORY_NUMERIC_DIM,
) -> Dict[str, torch.Tensor]:
    return {
        "trajectory_label_ids": torch.zeros(num_objects, dtype=torch.long),
        "trajectory_direction_ids": torch.zeros(num_objects, dtype=torch.long),
        "trajectory_numeric_feats": torch.zeros(num_objects, numeric_dim, dtype=torch.float32),
        "trajectory_object_mask": torch.zeros(num_objects, dtype=torch.long),
    }


@dataclass
class TrajectorySource:
    source_file: str
    lookup: Dict[Tuple[str, int], List[Dict]]
    label_to_id: Dict[str, int]
    direction_to_id: Dict[str, int]
    num_objects: int = TRAJECTORY_NUM_OBJECTS
    numeric_dim: int = TRAJECTORY_NUMERIC_DIM

    @classmethod
    def from_file(cls, source_file: str):
        resolved_path = resolve_trajectory_source_path(source_file)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Trajectory canonical file not found: {resolved_path}. "
                "Please create the sorted top-6 JSON first."
            )

        if resolved_path.endswith(".jsonl"):
            records = []
            with open(resolved_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            records = _normalize_flat_jsonl_records(records)
        else:
            with open(resolved_path, "r", encoding="utf-8") as f:
                records = json.load(f)

        if not isinstance(records, list):
            raise ValueError("Trajectory canonical JSON must be a list of records.")

        lookup: Dict[Tuple[str, int], List[Dict]] = {}
        labels = set()
        directions = set()

        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Trajectory record #{idx} must be an object.")
            for field in ("folder_id", "frame_id", "objects"):
                if field not in record:
                    raise ValueError(f"Trajectory record #{idx} is missing required field: {field}")
            folder_id = str(record["folder_id"])
            frame_id = int(record["frame_id"])
            objects = record["objects"]
            if not isinstance(objects, list):
                raise ValueError(f"Trajectory record #{idx} field 'objects' must be a list.")
            if len(objects) > TRAJECTORY_NUM_OBJECTS:
                raise ValueError(
                    f"Trajectory record #{idx} has {len(objects)} objects; max is {TRAJECTORY_NUM_OBJECTS}."
                )
            normalized_objects = []
            for obj_idx, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    raise ValueError(f"Trajectory object #{obj_idx} in record #{idx} must be an object.")
                missing = [field for field in ("label", "relative_position", "movement_angle", "speed_percent") if field not in obj]
                if missing:
                    raise ValueError(
                        f"Trajectory object #{obj_idx} in record #{idx} is missing required fields: {missing}"
                    )
                normalized = _normalize_object_payload(obj, key=(folder_id, frame_id), movement_is_degrees=False)
                normalized_objects.append(normalized)
                labels.add(str(normalized["label"]))
                directions.add(str(normalized["relative_position"]))
            key = (folder_id, frame_id)
            if key in lookup:
                raise ValueError(f"Duplicate trajectory key detected: {key}")
            lookup[key] = normalized_objects

        label_to_id = {TRAJECTORY_PAD_UNK_LABEL: TRAJECTORY_PAD_UNK_ID}
        for offset, label in enumerate(sorted(labels), start=1):
            label_to_id[label] = offset
        direction_to_id = {TRAJECTORY_PAD_UNK_LABEL: TRAJECTORY_PAD_UNK_ID}
        for offset, direction in enumerate(sorted(directions), start=1):
            direction_to_id[direction] = offset

        return cls(
            source_file=resolved_path,
            lookup=lookup,
            label_to_id=label_to_id,
            direction_to_id=direction_to_id,
            num_objects=TRAJECTORY_NUM_OBJECTS,
            numeric_dim=TRAJECTORY_NUMERIC_DIM,
        )

    def has_record(self, folder_id: str, frame_id: int) -> bool:
        return (str(folder_id), int(frame_id)) in self.lookup

    def encode(self, folder_id: str, frame_id: int) -> Dict[str, torch.Tensor]:
        objects = self.lookup.get((str(folder_id), int(frame_id)))
        if objects is None:
            return _empty_encoded_sample(self.num_objects, self.numeric_dim)

        label_ids = torch.zeros(self.num_objects, dtype=torch.long)
        direction_ids = torch.zeros(self.num_objects, dtype=torch.long)
        numeric_feats = torch.zeros(self.num_objects, self.numeric_dim, dtype=torch.float32)
        object_mask = torch.zeros(self.num_objects, dtype=torch.long)

        for slot, obj in enumerate(objects[: self.num_objects]):
            label_ids[slot] = self.label_to_id.get(str(obj["label"]), TRAJECTORY_PAD_UNK_ID)
            direction_ids[slot] = self.direction_to_id.get(str(obj["relative_position"]), TRAJECTORY_PAD_UNK_ID)
            numeric_feats[slot] = torch.tensor(
                [
                    float(obj["x1"]),
                    float(obj["y1"]),
                    float(obj["x2"]),
                    float(obj["y2"]),
                    float(obj["movement_angle"]),
                    float(obj["speed_percent"]),
                ],
                dtype=torch.float32,
            )
            object_mask[slot] = 1

        return {
            "trajectory_label_ids": label_ids,
            "trajectory_direction_ids": direction_ids,
            "trajectory_numeric_feats": numeric_feats,
            "trajectory_object_mask": object_mask,
        }


class TrajectoryBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        direction_vocab_size: int,
        d_cat: int = 32,
        d_dir: int = 16,
        d_numeric_hidden: int = 64,
        d_traj: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        num_objects: int = TRAJECTORY_NUM_OBJECTS,
    ):
        super().__init__()
        self.num_objects = num_objects
        self.d_traj = d_traj
        self.label_embedding = nn.Embedding(vocab_size, d_cat, padding_idx=TRAJECTORY_PAD_UNK_ID)
        self.direction_embedding = nn.Embedding(direction_vocab_size, d_dir, padding_idx=TRAJECTORY_PAD_UNK_ID)
        self.numeric_mlp = nn.Sequential(
            nn.Linear(TRAJECTORY_NUMERIC_DIM, d_numeric_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_numeric_hidden, d_numeric_hidden),
        )
        self.object_mlp = nn.Sequential(
            nn.Linear(d_cat + d_dir + d_numeric_hidden, d_traj),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_traj, d_traj),
        )
        self.slot_embedding = nn.Parameter(torch.zeros(1, num_objects, d_traj))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_traj,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=dropout,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, label_ids: torch.Tensor, direction_ids: torch.Tensor, numeric_feats: torch.Tensor, object_mask: torch.Tensor):
        label_embeds = self.label_embedding(label_ids)
        direction_embeds = self.direction_embedding(direction_ids)
        numeric_embeds = self.numeric_mlp(numeric_feats)
        tokens = self.object_mlp(torch.cat([label_embeds, direction_embeds, numeric_embeds], dim=-1))

        mask = object_mask.to(tokens.dtype).unsqueeze(-1)
        tokens = (tokens + self.slot_embedding) * mask

        key_padding_mask = object_mask == 0
        all_empty = key_padding_mask.all(dim=1)
        if all_empty.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_empty, 0] = False

        tokens = self.set_encoder(tokens, src_key_padding_mask=key_padding_mask)
        tokens = tokens * mask
        return tokens


class TrajectoryCLSHead(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 1024, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cls_query = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, traj_tokens: torch.Tensor, object_mask: torch.Tensor):
        batch_size = traj_tokens.shape[0]
        query = self.cls_query.expand(batch_size, -1, -1)
        key_padding_mask = object_mask == 0
        all_empty = key_padding_mask.all(dim=1)
        if all_empty.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_empty, 0] = False
        pooled, _ = self.cross_attn(query, traj_tokens, traj_tokens, key_padding_mask=key_padding_mask)
        output = self.out_proj(pooled)
        if all_empty.any():
            output = output.clone()
            output[all_empty] = 0
        return output


class TrajectoryConcatHead(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 896, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, traj_tokens: torch.Tensor):
        return self.proj(traj_tokens)


def attach_trajectory_branch(
    model,
    config: Dict,
    *,
    pixel_shuffle_dim: int,
    llm_hidden_size: int,
    logger=None,
):
    traj_cfg = get_trajectory_config(config)
    if not traj_cfg["enabled"]:
        model.trajectory_enabled = False
        return model

    if not getattr(model, "qformer_enabled", False):
        raise ValueError("Trajectory branch requires qformer.enabled=true in this v1 implementation.")

    fusion_mode = str(traj_cfg["fusion_mode"])
    if fusion_mode != "concat":
        raise ValueError(
            f"Unsupported trajectory fusion_mode for concat-only public repo: {fusion_mode}"
        )

    source = TrajectorySource.from_file(str(traj_cfg["source_file"]))
    model.trajectory_enabled = True
    model.trajectory_fusion_mode = fusion_mode
    model.trajectory_source_file = source.source_file
    model.trajectory_num_objects = int(traj_cfg["num_objects"])
    model.trajectory_vocab = dict(source.label_to_id)
    model.trajectory_direction_vocab = dict(source.direction_to_id)
    model.trajectory_qformer_token_count = int(getattr(model, "qformer_num_query_tokens", model.num_image_token))

    model.trajectory_backbone = TrajectoryBackbone(
        vocab_size=len(source.label_to_id),
        direction_vocab_size=len(source.direction_to_id),
        d_cat=int(traj_cfg["d_cat"]),
        d_dir=int(traj_cfg["d_dir"]),
        d_numeric_hidden=int(traj_cfg["d_numeric_hidden"]),
        d_traj=int(traj_cfg["d_traj"]),
        num_heads=int(traj_cfg["num_heads"]),
        num_layers=int(traj_cfg["num_layers"]),
        ffn_dim=int(traj_cfg["ffn_dim"]),
        dropout=float(traj_cfg["dropout"]),
        num_objects=int(traj_cfg["num_objects"]),
    ).to(dtype=torch.float32)
    model.trajectory_token_projector = TrajectoryConcatHead(
        input_dim=int(traj_cfg["d_traj"]),
        output_dim=llm_hidden_size,
        dropout=float(traj_cfg["dropout"]),
    ).to(dtype=torch.float32)
    model.trajectory_cls_head = None
    model.num_image_token = model.trajectory_qformer_token_count + int(traj_cfg["num_objects"])

    model._trajectory_label_ids = None
    model._trajectory_direction_ids = None
    model._trajectory_numeric_feats = None
    model._trajectory_object_mask = None
    model.set_trajectory_inputs = _set_trajectory_inputs.__get__(model)
    model.clear_trajectory_inputs = _clear_trajectory_inputs.__get__(model)
    model.get_trajectory_inputs = _get_trajectory_inputs.__get__(model)

    if logger:
        logger.info(
            "Attached trajectory branch: "
            f"fusion_mode={fusion_mode}, source_file={source.source_file}, "
            f"vocab_size={len(source.label_to_id)}, num_image_token={model.num_image_token}"
        )
    return model


def _set_trajectory_inputs(self, label_ids: torch.Tensor, direction_ids: torch.Tensor, numeric_feats: torch.Tensor, object_mask: torch.Tensor):
    self._trajectory_label_ids = label_ids
    self._trajectory_direction_ids = direction_ids
    self._trajectory_numeric_feats = numeric_feats
    self._trajectory_object_mask = object_mask


def _clear_trajectory_inputs(self):
    self._trajectory_label_ids = None
    self._trajectory_direction_ids = None
    self._trajectory_numeric_feats = None
    self._trajectory_object_mask = None


def _get_trajectory_inputs(self, batch_size: int, device: torch.device):
    label_ids = getattr(self, "_trajectory_label_ids", None)
    direction_ids = getattr(self, "_trajectory_direction_ids", None)
    numeric_feats = getattr(self, "_trajectory_numeric_feats", None)
    object_mask = getattr(self, "_trajectory_object_mask", None)
    if label_ids is None or direction_ids is None or numeric_feats is None or object_mask is None:
        empty = _empty_encoded_sample()
        label_ids = empty["trajectory_label_ids"].unsqueeze(0).repeat(batch_size, 1)
        direction_ids = empty["trajectory_direction_ids"].unsqueeze(0).repeat(batch_size, 1)
        numeric_feats = empty["trajectory_numeric_feats"].unsqueeze(0).repeat(batch_size, 1, 1)
        object_mask = empty["trajectory_object_mask"].unsqueeze(0).repeat(batch_size, 1)
    return (
        label_ids.to(device=device),
        direction_ids.to(device=device),
        numeric_feats.to(device=device, dtype=torch.float32),
        object_mask.to(device=device),
    )


def build_trajectory_tokens_base(model, batch_size: int, device: torch.device):
    label_ids, direction_ids, numeric_feats, object_mask = model.get_trajectory_inputs(batch_size, device)
    return model.trajectory_backbone(
        label_ids=label_ids,
        direction_ids=direction_ids,
        numeric_feats=numeric_feats,
        object_mask=object_mask,
    ), object_mask


def build_trajectory_features(model, batch_size: int, device: torch.device):
    traj_tokens_base, object_mask = build_trajectory_tokens_base(model, batch_size, device)
    if model.trajectory_fusion_mode != "concat":
        raise ValueError(
            f"Unsupported trajectory fusion_mode for concat-only public repo: {model.trajectory_fusion_mode}"
        )
    projected = model.trajectory_token_projector(traj_tokens_base)
    return projected * object_mask.to(projected.dtype).unsqueeze(-1)


def _trajectory_state_dict(model) -> Dict[str, torch.Tensor]:
    state = {}
    for module_name in ("trajectory_backbone", "trajectory_token_projector"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for key, value in module.state_dict().items():
            state[f"{module_name}.{key}"] = value.detach().cpu()
    return state


def save_trajectory_branch(model, output_dir: str):
    if not getattr(model, "trajectory_enabled", False):
        return
    os.makedirs(output_dir, exist_ok=True)
    save_file(_trajectory_state_dict(model), os.path.join(output_dir, TRAJECTORY_BRANCH_WEIGHTS_NAME))
    metadata = {
        "enabled": True,
        "fusion_mode": model.trajectory_fusion_mode,
        "source_file": model.trajectory_source_file,
        "num_objects": getattr(model, "trajectory_num_objects", TRAJECTORY_NUM_OBJECTS),
        "qformer_token_count": getattr(model, "trajectory_qformer_token_count", None),
        "num_image_token": getattr(model, "num_image_token", None),
        "stage": getattr(model, "pretrain_stage", getattr(model, "training_stage", None)),
        "movement_enabled": getattr(model, "pretrain_movement_enabled", None),
        "pretrain_data_source": getattr(model, "pretrain_data_source", None),
        "question_format_version": getattr(model, "question_format_version", None),
    }
    with open(os.path.join(output_dir, TRAJECTORY_BRANCH_CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def read_trajectory_branch_metadata(checkpoint_dir: str) -> Dict:
    config_path = os.path.join(checkpoint_dir, TRAJECTORY_BRANCH_CONFIG_NAME)
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_trajectory_branch(model, checkpoint_dir: str, strict: bool = True):
    weights_path = os.path.join(checkpoint_dir, TRAJECTORY_BRANCH_WEIGHTS_NAME)
    config_path = os.path.join(checkpoint_dir, TRAJECTORY_BRANCH_CONFIG_NAME)
    if not os.path.exists(weights_path):
        if strict:
            raise FileNotFoundError(f"Trajectory branch weights not found: {weights_path}")
        return False

    metadata = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    checkpoint_mode = metadata.get("fusion_mode")
    current_mode = getattr(model, "trajectory_fusion_mode", None)
    if checkpoint_mode and current_mode and checkpoint_mode != current_mode:
        raise ValueError(
            f"Trajectory fusion mode mismatch: checkpoint={checkpoint_mode}, current={current_mode}"
        )

    state = load_file(weights_path, device="cpu")
    for module_name in ("trajectory_backbone", "trajectory_token_projector"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        module_state = {
            key[len(module_name) + 1 :]: value
            for key, value in state.items()
            if key.startswith(module_name + ".")
        }
        try:
            module.load_state_dict(module_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Trajectory checkpoint architecture mismatch while loading "
                f"{module_name}. Check fusion_mode/d_traj/num_layers/ffn_dim/dropout/code version. "
                f"Checkpoint dir: {checkpoint_dir}. Original error: {exc}"
            ) from exc
    return True
