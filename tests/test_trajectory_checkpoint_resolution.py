from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_train_checkpoint_download_patterns_include_trajectory_artifacts():
    content = _read("train.py")
    assert '"trajectory_branch.safetensors"' in content
    assert '"trajectory_branch_config.json"' in content


def test_pretrain_checkpoint_verifier_reads_trajectory_artifacts():
    content = _read("pretrain_checkpoint_verify.py")
    assert "TRAJECTORY_BRANCH_WEIGHTS_NAME" in content
    assert "load_file(trajectory_path, device=\"cpu\")" in content
