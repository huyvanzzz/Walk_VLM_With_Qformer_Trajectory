from gptscore.constants import ALLOWED_LABELS, CRITERION_KEYS


def validate_input_pair(pair):
    errors = []
    if "ground_truth" not in pair:
        errors.append("Missing required field: ground_truth")
    elif not str(pair["ground_truth"]).strip():
        errors.append("ground_truth must be non-empty")

    if "generation" not in pair:
        errors.append("Missing required field: generation")
    elif not str(pair["generation"]).strip():
        errors.append("generation must be non-empty")

    return errors


def validate_judge_output(payload):
    errors = []
    if not isinstance(payload, dict):
        return ["Judge output must be a JSON object"]

    criteria = payload.get("criteria")
    if not isinstance(criteria, dict):
        errors.append("Missing or invalid criteria object")
        return errors

    for key in CRITERION_KEYS:
        criterion = criteria.get(key)
        if not isinstance(criterion, dict):
            errors.append(f"Missing criterion object: {key}")
            continue
        if "applicable" not in criterion or not isinstance(criterion["applicable"], bool):
            errors.append(f"{key}: applicable must be a bool")
        if "label" not in criterion:
            errors.append(f"{key}: missing label")
        else:
            label = criterion["label"]
            applicable = criterion.get("applicable")
            if applicable is False and label is not None:
                errors.append(f"{key}: non-applicable criterion must use label=null")
            elif applicable is True and label is None:
                errors.append(f"{key}: applicable criterion must not use label=null")
            elif label is not None and label not in ALLOWED_LABELS:
                errors.append(f"{key}: label must be one of {sorted(ALLOWED_LABELS)} or null")
        if "rationale" not in criterion or not isinstance(criterion["rationale"], str):
            errors.append(f"{key}: rationale must be a string")

    if "overall_rationale" not in payload or not isinstance(payload["overall_rationale"], str):
        errors.append("overall_rationale must be a string")

    return errors

