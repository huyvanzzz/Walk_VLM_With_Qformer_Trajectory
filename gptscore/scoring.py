from collections import Counter

from gptscore.constants import LABEL_TO_SCORE
from gptscore.validation import validate_judge_output


def score_judge_output(judge_output):
    errors = validate_judge_output(judge_output)
    if errors:
        raise ValueError("; ".join(errors))

    applicable_scores = []
    for criterion in judge_output["criteria"].values():
        if criterion["applicable"]:
            applicable_scores.append(LABEL_TO_SCORE[criterion["label"]])

    if not applicable_scores:
        raise ValueError("No applicable criteria available for scoring")

    overall_score = sum(applicable_scores) / len(applicable_scores)

    return {
        "overall_score": round(overall_score, 6),
    }


def score_judged_document(judged_doc):
    items = judged_doc.get("items", [])
    scored_items = []
    skipped_items = []
    judge_status_counts = Counter()
    for item in items:
        status = item.get("judge_status", "unknown")
        judge_status_counts[status] += 1

        if status != "scored":
            skipped_items.append(item)
            continue

        judge_output = item.get("judge_output")
        if judge_output is None:
            skipped = dict(item)
            skipped["validation_status"] = "missing_judge_output"
            skipped_items.append(skipped)
            continue

        try:
            sample_score = score_judge_output(judge_output)
        except ValueError as exc:
            skipped = dict(item)
            skipped["validation_status"] = "score_validation_failed"
            skipped["error_message"] = str(exc)
            skipped_items.append(skipped)
            continue

        enriched = dict(item)
        enriched["sample_score"] = sample_score
        scored_items.append(enriched)

    valid_scores = [item["sample_score"]["overall_score"] for item in scored_items]
    mean_over_valid_samples = (
        round(sum(valid_scores) / len(valid_scores), 6) if valid_scores else None
    )

    return {
        "summary": {
            "total_items": len(items),
            "scored_items": len(scored_items),
            "skipped_items": len(skipped_items),
            "judge_status_counts": dict(judge_status_counts),
            "mean_over_valid_samples": mean_over_valid_samples,
        },
        "scored_items": scored_items,
        "skipped_items": skipped_items,
    }
