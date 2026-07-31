import argparse
import sys
import yaml

sys.path.append(".")
from qformer_bridge import get_qformer_config, _load_qformer_from_source


def main():
    parser = argparse.ArgumentParser(description="Download and validate Q-Former assets.")
    parser.add_argument("--config", default="internvl_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    q_cfg = get_qformer_config(config)
    if not q_cfg["enabled"]:
        print("Q-Former is disabled in config. Nothing to prepare.")
        return

    print(f"Preparing Q-Former from: {q_cfg['source_model']}")
    qformer, query_tokens, tokenizer, blip_config = _load_qformer_from_source(
        q_cfg["source_model"],
        q_cfg["cache_dir"],
    )
    print("Q-Former ready.")
    print(f"  hidden_size: {blip_config.qformer_config.hidden_size}")
    print(f"  encoder_hidden_size: {blip_config.qformer_config.encoder_hidden_size}")
    print(f"  checkpoint query_tokens: {query_tokens.shape[1]}")
    print(f"  tokenizer vocab size: {len(tokenizer)}")
    print(f"  qformer params: {sum(p.numel() for p in qformer.parameters()):,}")


if __name__ == "__main__":
    main()
