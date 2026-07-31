import json
from pathlib import Path

import torch

from qformer_bridge import BRIDGE_CONFIG_NAME, save_qformer_bridge
from trajectory_branch import TRAJECTORY_BRANCH_CONFIG_NAME, save_trajectory_branch


class _TinyBridge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qformer_enabled = True
        self.qformer_source_model = "dummy-source"
        self.qformer_num_query_tokens = 32
        self.num_image_token = 38
        self.qformer_input_proj = torch.nn.Linear(4, 4)
        self.qformer_to_mlp1_proj = torch.nn.Linear(4, 4)
        self.pretrain_stage = "pretrain"
        self.pretrain_data_source = "./json/question_train.jsonl"
        self.question_format_version = "v1_qa_question_answer"
        self.pretrain_movement_enabled = True


class _TinyTrajectory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from trajectory_branch import TrajectoryBackbone, TrajectoryConcatHead

        self.trajectory_enabled = True
        self.trajectory_fusion_mode = "concat"
        self.trajectory_source_file = "json/results_botsort_top6_sorted.jsonl"
        self.trajectory_num_objects = 6
        self.trajectory_qformer_token_count = 32
        self.num_image_token = 38
        self.trajectory_backbone = TrajectoryBackbone(vocab_size=5, direction_vocab_size=5)
        self.trajectory_cls_head = None
        self.trajectory_token_projector = TrajectoryConcatHead(output_dim=896)
        self.pretrain_stage = "pretrain"
        self.pretrain_data_source = "./json/question_train.jsonl"
        self.question_format_version = "v1_qa_question_answer"
        self.pretrain_movement_enabled = False


def test_pretrain_bridge_metadata_contains_handoff_fields(tmp_path):
    model = _TinyBridge()
    save_qformer_bridge(model, str(tmp_path))

    metadata = json.loads((tmp_path / BRIDGE_CONFIG_NAME).read_text(encoding="utf-8"))

    assert metadata["stage"] == "pretrain"
    assert metadata["movement_enabled"] is True
    assert metadata["pretrain_data_source"] == "./json/question_train.jsonl"
    assert metadata["question_format_version"] == "v1_qa_question_answer"


def test_pretrain_trajectory_metadata_contains_handoff_fields(tmp_path):
    model = _TinyTrajectory()
    save_trajectory_branch(model, str(tmp_path))

    metadata = json.loads((tmp_path / TRAJECTORY_BRANCH_CONFIG_NAME).read_text(encoding="utf-8"))

    assert metadata["stage"] == "pretrain"
    assert metadata["fusion_mode"] == "concat"
    assert metadata["movement_enabled"] is False
    assert metadata["pretrain_data_source"] == "./json/question_train.jsonl"
    assert metadata["question_format_version"] == "v1_qa_question_answer"


def test_train_py_defines_separate_pretrain_checkpoint_interface():
    content = Path("train.py").read_text(encoding="utf-8")

    assert '--pretrain_checkpoint' in content
    assert 'if args.checkpoint and args.pretrain_checkpoint' in content
    assert 'Loading LoRA adapter from checkpoint' in content


def test_train_pretrain_supports_resume_checkpoint_surface():
    content = Path("train_pretrain.py").read_text(encoding="utf-8")

    assert '--checkpoint' in content or '--resume_checkpoint' in content
    assert 'optimizer.pt' in content
    assert 'scheduler.pt' in content
    assert 'training_state.json' in content
    assert 'early_stopping_state.json' in content


def test_concat_only_runtime_surfaces_are_declared_in_code():
    train_content = Path("train.py").read_text(encoding="utf-8")
    pretrain_content = Path("train_pretrain.py").read_text(encoding="utf-8")
    branch_content = Path("trajectory_branch.py").read_text(encoding="utf-8")

    assert "Unsupported trajectory fusion_mode for concat-only public repo" in branch_content
    assert "--pretrain_checkpoint" in train_content
    assert "load_qformer_bridge(model, pretrain_checkpoint_dir, strict=True)" in train_content
    assert "load_trajectory_branch(model, pretrain_checkpoint_dir, strict=True)" in train_content
    assert "validate_pretrain_resume_checkpoint" in pretrain_content
