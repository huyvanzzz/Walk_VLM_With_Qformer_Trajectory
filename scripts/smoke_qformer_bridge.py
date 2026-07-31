import argparse
import sys

import torch
import yaml
from transformers import AutoModel, BitsAndBytesConfig

sys.path.append(".")
from qformer_bridge import attach_qformer_bridge, qformer_enabled, trainable_parameter_summary


def main():
    parser = argparse.ArgumentParser(description="Smoke test InternVL Q-Former bridge.")
    parser.add_argument("--config", default="internvl_config.yaml")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not qformer_enabled(config):
        raise SystemExit("Q-Former is disabled; smoke test skipped.")

    quant_cfg = None
    if not args.cpu and config["model"]["quantization"]["enabled"]:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=config["model"]["quantization"]["double_quant"],
            bnb_4bit_quant_type=config["model"]["quantization"]["type"],
        )

    model = AutoModel.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.bfloat16 if not args.cpu else torch.float32,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    attach_qformer_bridge(model, config)
    model.vision_model.requires_grad_(False)
    model.eval()

    device = torch.device("cpu" if args.cpu else "cuda")
    if args.cpu:
        model.to(device)

    pixel_values = torch.zeros(
        1,
        3,
        448,
        448,
        dtype=torch.float32 if args.cpu else torch.bfloat16,
        device=device,
    )
    q_ids, q_mask = model.encode_qformer_texts(
        ["Analyze the scene and give guidance for a visually impaired user."],
        device=device,
    )
    model.set_qformer_text(q_ids, q_mask)
    with torch.no_grad():
        visual_embeds = model.extract_feature(pixel_values)
    model.clear_qformer_text()

    expected_tokens = model.num_image_token
    assert visual_embeds.shape[1] == expected_tokens, visual_embeds.shape
    assert visual_embeds.shape[-1] == model.config.llm_config.hidden_size, visual_embeds.shape

    print("Smoke test passed.")
    print(f"  visual_embeds: {tuple(visual_embeds.shape)}")
    print(f"  num_image_token: {model.num_image_token}")
    print("Trainable parameters:")
    for row in trainable_parameter_summary(model):
        print(f"  {row}")


if __name__ == "__main__":
    main()
