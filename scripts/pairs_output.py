import json
import os


def write_prediction_pairs(output_file, checkpoint_label, split_name, detailed_results):
    output_dir = os.path.dirname(output_file) or "."
    stem = os.path.splitext(os.path.basename(output_file))[0]
    pairs_path = os.path.join(output_dir, f"{stem}_pairs.json")
    pairs_output = {
        "checkpoint": checkpoint_label,
        "split": split_name,
        "pairs": [
            {
                "id": item["id"],
                "ground_truth": item["ground_truth"],
                "generation": item["prediction"],
            }
            for item in detailed_results
        ],
    }
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(pairs_output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Đã lưu cặp GT/Generation tại: {pairs_path}")
    return pairs_path
