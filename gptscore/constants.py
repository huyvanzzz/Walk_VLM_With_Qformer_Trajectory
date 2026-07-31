ALLOWED_LABELS = {"Fail", "Weak", "Acceptable", "Strong"}

CRITERION_KEYS = [
    "safety_correctness",
    "hazard_path_state_fidelity",
    "direction_fidelity",
    "action_usefulness",
    "spoken_guidance_quality",
]

LABEL_TO_SCORE = {
    "Fail": 0,
    "Weak": 1,
    "Acceptable": 2,
    "Strong": 3,
}

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
SCHEMA_VERSION = "gptscore-five-criteria-v1"
