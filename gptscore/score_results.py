import argparse
from pathlib import Path

from gptscore.io_utils import load_pairs_file, save_json
from gptscore.scoring import score_judged_document


def parse_args():
    parser = argparse.ArgumentParser(description="Score a judged GPTScore JSON file")
    parser.add_argument("--input", required=True, help="Path to *_gptscore_judged.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def default_output_path(input_path):
    path = Path(input_path)
    stem = path.stem
    if stem.endswith("_gptscore_judged"):
        stem = stem[:-16]
    return str(path.with_name(f"{stem}_gptscore_scored.json"))


def main():
    args = parse_args()
    judged_doc = load_pairs_file(args.input)
    result = score_judged_document(judged_doc)
    result["input_file"] = str(Path(args.input).resolve())
    output_path = args.output or default_output_path(args.input)
    save_json(output_path, result)
    print(f"Saved scored results to: {output_path}")


if __name__ == "__main__":
    main()

