from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_gptscore_is_five_criteria_only():
    constants = _read("gptscore/constants.py")
    validation = _read("gptscore/validation.py")
    scoring = _read("gptscore/scoring.py")

    assert "safety_correctness" in constants
    assert "hazard_path_state_fidelity" in constants
    assert "direction_fidelity" in constants
    assert "action_usefulness" in constants
    assert "spoken_guidance_quality" in constants

    assert "GATE_KEYS" not in constants
    assert "SIGNAL_KEYS" not in constants
    assert "signals_in_gt" not in validation
    assert '"gate"' not in scoring
    assert "applied_gate_cap" not in scoring


def test_gptscore_outputs_do_not_expose_old_method_metadata():
    providers = _read("gptscore/providers.py")
    judge_runner = _read("gptscore/judge_runner.py")
    run_judge = _read("gptscore/run_judge.py")

    assert "restore-779-gptscore" not in providers
    assert ".env.example" not in providers
    assert "checkpoint" not in judge_runner
    assert "prompt_version" not in judge_runner
    assert "schema_version" not in judge_runner
    assert "prompt_profile" not in run_judge
