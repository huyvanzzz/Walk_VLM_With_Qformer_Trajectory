import argparse
from pathlib import Path

from gptscore.io_utils import load_pairs_file, save_json
from gptscore.judge_runner import judge_pairs_document
from gptscore.providers import make_provider_judge_callable


def parse_args():
    parser = argparse.ArgumentParser(description="Run GPTScore judge over pairs.json")
    parser.add_argument("--input", required=True, help="Path to *_pairs.json")
    parser.add_argument("--provider", choices=["openai", "openrouter"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="head")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=1)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=90)
    return parser.parse_args()


def default_output_path(input_path):
    path = Path(input_path)
    stem = path.stem
    if stem.endswith("_pairs"):
        stem = stem[:-6]
    return str(path.with_name(f"{stem}_gptscore_judged.json"))


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    pairs_doc = load_pairs_file(args.input)
    judge_callable, resolved_model = make_provider_judge_callable(
        provider=args.provider,
        model=args.model,
        repo_root=repo_root,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    judged = judge_pairs_document(
        pairs_doc,
        provider=args.provider,
        model=resolved_model,
        judge_callable=judge_callable,
        limit=args.limit,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        preview_count=args.preview_count,
        show_progress=not args.disable_progress,
    )
    judged["input_file"] = str(Path(args.input).resolve())
    output_path = args.output or default_output_path(args.input)
    save_json(output_path, judged)
    print(f"Saved judged results to: {output_path}")


if __name__ == "__main__":
    main()
