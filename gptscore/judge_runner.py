from datetime import datetime, timezone

from gptscore.io_utils import select_pair_items
from gptscore.validation import validate_input_pair, validate_judge_output


def _default_emit(message):
    print(message)


def _default_progress_factory(iterable, total=None, desc=None):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _emit_preview(selected_pairs, preview_count, emit):
    if preview_count <= 0 or not selected_pairs:
        return
    emit(f"Previewing {min(preview_count, len(selected_pairs))} selected sample(s):")
    for pair in selected_pairs[:preview_count]:
        emit(f"  id={pair.get('id')}")
        emit(f"  ground_truth: {pair.get('ground_truth')}")
        emit(f"  generation: {pair.get('generation')}")


def judge_pairs_document(
    pairs_doc,
    provider,
    model,
    judge_callable,
    limit=None,
    sample_mode="head",
    sample_seed=0,
    preview_count=1,
    emit=None,
    show_progress=True,
    progress_factory=None,
):
    selected_pairs = select_pair_items(
        pairs_doc,
        limit=limit,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
    )
    emit = emit or _default_emit
    progress_factory = progress_factory or _default_progress_factory

    judged = {
        "provider": provider,
        "model": model,
        "split": pairs_doc.get("split"),
        "total_input_pairs": len(pairs_doc.get("pairs", [])),
        "selected_count": len(selected_pairs),
        "sample_mode": sample_mode,
        "sample_seed": sample_seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
    }

    _emit_preview(selected_pairs, preview_count, emit)

    iterable = enumerate(selected_pairs)
    if show_progress:
        iterable = progress_factory(iterable, total=len(selected_pairs), desc="Judging pairs")

    for index, pair in iterable:
        item = {
            "id": pair.get("id", index),
            "ground_truth": pair.get("ground_truth"),
            "generation": pair.get("generation"),
            "judge_status": None,
            "raw_response": None,
            "judge_output": None,
            "error_type": None,
            "error_message": None,
        }

        input_errors = validate_input_pair(pair)
        if input_errors:
            item["judge_status"] = "input_invalid"
            item["error_type"] = "input_invalid"
            item["error_message"] = "; ".join(input_errors)
            judged["items"].append(item)
            continue

        try:
            judge_output, raw_response = judge_callable(
                str(pair["ground_truth"]),
                str(pair["generation"]),
            )
            item["raw_response"] = raw_response
            item["judge_output"] = judge_output
        except RuntimeError as exc:
            item["judge_status"] = "transport_error"
            item["error_type"] = "transport_error"
            item["error_message"] = str(exc)
            judged["items"].append(item)
            continue
        except ValueError as exc:
            item["judge_status"] = "parse_failed"
            item["error_type"] = "parse_failed"
            item["error_message"] = str(exc)
            judged["items"].append(item)
            continue

        output_errors = validate_judge_output(judge_output)
        if output_errors:
            item["judge_status"] = "schema_invalid"
            item["error_type"] = "schema_invalid"
            item["error_message"] = "; ".join(output_errors)
        else:
            item["judge_status"] = "scored"

        judged["items"].append(item)

    return judged
