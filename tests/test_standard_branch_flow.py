from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_standard_branch_prompt_flow_is_last_frame_direct_text_only():
    wad_dataset = _read("wad_dataset.py")
    preprocessing = _read("preprocessing.py")
    data_module = _read("data.py")
    train_module = _read("train.py")

    assert "Chain-of-Thought" not in wad_dataset
    assert "COT" not in wad_dataset
    assert "structured_json" not in wad_dataset
    assert 'response_format: str = "direct_text"' in wad_dataset
    assert 'self.response_format = "direct_text"' in wad_dataset
    assert "Describe the scene for a visually impaired user based on the final frame." in wad_dataset
    assert "Provide only the final spoken guidance in natural language." in wad_dataset
    assert "last_frame_id = frame_ids[-1]" in wad_dataset
    assert "self._load_frames(frame_path, [last_frame_id])" in wad_dataset

    assert 'return "direct_text"' in preprocessing
    assert 'response_format: str = "direct_text"' in preprocessing
    assert "structured_json" not in preprocessing
    assert "<answer>" not in preprocessing

    assert "self.task_prompt" not in data_module
    assert "internvl_config_traj_concat.yaml" not in train_module
    assert "with open(CONFIG_PATH" not in train_module
    assert 'parser.add_argument("--config", type=str, required=True' in train_module


def test_gptscore_provider_is_single_neutral_path():
    providers = _read("gptscore/providers.py")

    assert "PROMPT_PROFILE_TO_FILE" not in providers
    assert "prompt_profile" not in providers
    assert "prompt_version" not in providers
    assert "schemas_dir" not in providers
    assert "prompts_dir" not in providers
    assert "gptscore_alter" not in providers
