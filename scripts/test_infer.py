import os
import yaml
import json
import torch
import argparse
import pickle
from huggingface_hub import snapshot_download
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import sys
sys.path.append('.')
# Import class Metrics từ file metrics.py
from scripts.metrics import VLMMetrics
# Import các thành phần data từ project của bạn
from wad_dataset import WADDatasetForInternVL
from preprocessing import get_response_format
from qformer_bridge import attach_qformer_bridge, load_qformer_bridge, qformer_enabled
from model.conversation import get_conv_template
from scripts.pairs_output import write_prediction_pairs
from trajectory_branch import build_trajectory_source_from_config, load_trajectory_branch

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
SYSTEM_MESSAGE = "You are a navigation assistant for visually impaired users."


def log_runtime_prompt_state(model, stage):
    print(
        f"[PROMPT STATE][{stage}] template={getattr(model, 'template', 'unknown')} | "
        f"system_message={repr(getattr(model, 'system_message', ''))}"
    )


def align_language_model_devices(model):
    target_device = torch.device("cuda:0")
    input_embeddings = model.language_model.get_input_embeddings()
    input_embeddings.to(device=target_device)
    output_embeddings = model.language_model.get_output_embeddings()
    if output_embeddings is not None:
        output_embeddings.to(device=target_device)
    embedding_device = next(input_embeddings.parameters()).device
    print(f"[DEVICE CHECK] input_embeddings device: {embedding_device}", flush=True)


def run_model_chat(model, tokenizer, pixel_values, question, generation_config):
    num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    embedding_device = model.language_model.get_input_embeddings().weight.device
    input_ids = model_inputs["input_ids"].to(embedding_device)
    attention_mask = model_inputs["attention_mask"].to(embedding_device)
    generation_config = dict(generation_config)
    generation_config["eos_token_id"] = eos_token_id
    generation_config["pad_token_id"] = eos_token_id

    generation_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config,
    )
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    return response.split(template.sep)[0].strip()

class TestCollaterFn:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
    
    def __call__(self, batch):
        return batch

def resolve_checkpoint_path(checkpoint):
    """Nếu checkpoint không phải local path, download từ HuggingFace về cache."""
    if not checkpoint:
        return None
    if os.path.exists(checkpoint):
        print(f"Using local checkpoint path: {checkpoint}")
        return checkpoint
    print(f"Checkpoint '{checkpoint}' không phải local path. Đang tải từ HuggingFace...")
    return snapshot_download(
        repo_id=checkpoint,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "qformer_bridge.safetensors",
            "qformer_bridge_config.json",
            "trajectory_branch.safetensors",
            "trajectory_branch_config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Test InternVL VLM Model")
    parser.add_argument("--config", type=str, default="internvl_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local checkpoint dir hoặc HuggingFace Repo ID")
    parser.add_argument("--split", type=str, default="test_QA", choices=["test_QA", "test_alter", "val"])
    parser.add_argument("--output_file", type=str, default="results/eval_results.json")
    parser.add_argument("--print_samples", type=int, default=5)
    return parser.parse_args()

def prepare_auxiliary_data(config):
    """Hàm phụ trợ load frame index và bbox cho tập test"""
    print("--- Loading Auxiliary Data for Testing ---")
    
    index_file = "./wad_dataset/frame_index.pkl"
    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            frame_index = pickle.load(f)
    else:
        raise FileNotFoundError(f"Frame index not found at {index_file}.")

    bbox_file = "all_bboxes_1.jsonl"
    if os.path.exists(bbox_file):
        bbox_dataset = load_dataset("json", data_files=bbox_file, split="train")
    else:
        bbox_dataset = load_dataset(config['data']['name'], data_files="all_bboxes_1.jsonl", split="train")

    bbox_by_folder = defaultdict(lambda: defaultdict(list))
    for bbox_entry in bbox_dataset:
        folder_id = bbox_entry['folder_id']
        frame_id = bbox_entry['frame_id']
        bbox_by_folder[folder_id][frame_id].append({
            'label': bbox_entry['label'],
            'confidence': bbox_entry['probs'],
            'bbox': bbox_entry['boxs'],
            'relative_position': bbox_entry.get('relative_position', "unknown"),
            'distance_zone': bbox_entry.get('distance_zone', 'unknown'),
            'coming_to_user': bbox_entry.get('coming_to_user', False),
            'speed': bbox_entry.get('speed', 0.0),
            'danger_score': bbox_entry.get('danger_score', 0.0)
        })
    trajectory_source = build_trajectory_source_from_config(config)
    return frame_index, bbox_by_folder, trajectory_source


def main():
    args = parse_args()
    # Resolve checkpoint: download từ HF nếu cần (trước mọi thao tác load)
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    response_format = get_response_format(config)

    # 1. Load Base Model & Tokenizer
    model_name_or_path = config['model']['name']
    batch_size = config['training']['batch_size']
        
    # 2. Cấu hình Quantization 4-bit
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['quantization']['enabled'],
        bnb_4bit_compute_dtype=torch.bfloat16 if config['model']['quantization']['compute_dtype'] == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=config['model']['quantization']['double_quant'],
        bnb_4bit_quant_type=config['model']['quantization']['type']
    )
    # 3. Load model
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=config['model']['trust_remote_code']
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    log_runtime_prompt_state(model, "after_load_before_override")
    model.system_message = SYSTEM_MESSAGE
    log_runtime_prompt_state(model, "after_override")
    if qformer_enabled(config):
        attach_qformer_bridge(model, config)
    model.eval()

    # 2. Load Checkpoint LoRA
    if args.checkpoint:
        print(f"Loading LoRA weights from: {args.checkpoint}")
        if qformer_enabled(config):
            load_qformer_bridge(model, args.checkpoint, strict=True)
            print("✓ Q-Former bridge loaded successfully.")
        if getattr(model, "trajectory_enabled", False):
            load_trajectory_branch(model, args.checkpoint, strict=True)
            print("✓ Trajectory branch loaded successfully.")
        model.language_model = PeftModel.from_pretrained(
            model.language_model, 
            args.checkpoint, 
            is_trainable=False,
            device_map={"": 0},
        )
        print("✓ LoRA Adapter loaded successfully.")
    else:
        print("No checkpoint provided. Evaluating Zero-shot (Base Model).")
    if not config['model']['quantization']['enabled']:
        model = model.cuda()
    align_language_model_devices(model)

    # ==========================================
    # 3. CHUẨN BỊ TẬP TEST CHÍNH XÁC THEO ARGUMENTS
    # ==========================================
    print(f"Building dataset for split: {args.split}...")
    
    frame_index, bbox_by_folder, trajectory_source = prepare_auxiliary_data(config)
    
    if args.split == "test_alter":
        data_file = "test_alter.json" 
    elif args.split == "test_QA":
        data_file = "test_QA.json"
        
    print(f"Loading metadata from {data_file}...")
    dataset_dict = load_dataset(
        config['data']['name'],
        data_files={
            "test": data_file
        }
    )

    total_samples = len(dataset_dict["test"])
    print(f"Loaded full test split with {total_samples} samples.")

    test_dataset = WADDatasetForInternVL(
        metadata_dataset=dataset_dict,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        trajectory_source=trajectory_source,
        split='test',
        response_format=response_format,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=TestCollaterFn(tokenizer, model), 
        shuffle=False
    )

    # 4. Evaluation Loop
    predictions, references, detailed_results = [], [], []

    print("\n" + "="*50)
    print(f" BẮT ĐẦU CHẠY ĐÁNH GIÁ TRÊN TẬP: {args.split} (Tổng: {len(test_dataset)} samples)")
    print("="*50)

    evaluator = VLMMetrics()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            sample = batch[0]
            pixel_values = torch.cat([torch.as_tensor(p) for p in sample['pixel_values']], dim=0).to(torch.bfloat16).cuda()
            question = str(sample['question'])
            ground_truth = str(sample['answer'])
            
            generation_config = dict(
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
                repetition_penalty=1.3,
                early_stopping=True,
            )
            
            if getattr(model, "qformer_enabled", False):
                qformer_text = sample.get("qformer_text", question.replace("<image>", "").strip())
                q_ids, q_mask = model.encode_qformer_texts(
                    [qformer_text] * pixel_values.shape[0],
                    device=pixel_values.device,
                )
                model.set_qformer_text(q_ids, q_mask)
            if getattr(model, "trajectory_enabled", False):
                model.set_trajectory_inputs(
                    sample["trajectory_label_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                    sample["trajectory_direction_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                    sample["trajectory_numeric_feats"].unsqueeze(0).repeat(pixel_values.shape[0], 1, 1).cuda(),
                    sample["trajectory_object_mask"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                )
            response = run_model_chat(model, tokenizer, pixel_values, question, generation_config)
            if getattr(model, "qformer_enabled", False):
                model.clear_qformer_text()
            if getattr(model, "trajectory_enabled", False):
                model.clear_trajectory_inputs()
            question_token_count = len(tokenizer.encode(question, add_special_tokens=False))
            response_token_count = len(tokenizer.encode(response, add_special_tokens=False))
            ground_truth_token_count = len(tokenizer.encode(ground_truth, add_special_tokens=False))
            
            predictions.append(response)
            references.append(ground_truth)
            
            if i < args.print_samples:
                print(f"\n--- Sample {i+1} ---")
                print(
                    f"Token stats | Q: {question_token_count} | "
                    f"Pred: {response_token_count} | GT: {ground_truth_token_count}"
                )
                print(f"Q: {question}")
                print(f"Pred: {response}")
                print(f"GT:   {ground_truth}")
            
            detailed_results.append({
                "id": i,
                "question": question,
                "prediction": response,
                "ground_truth": ground_truth
            })

    # 5. Compute Metrics
    metric_target_field = "raw_text" if response_format == "direct_text" else "instruction"
    print(f"\nComputing Metrics (ROUGE, TF-IDF) on '{metric_target_field}'...")
    metrics = evaluator.compute(predictions, references, target_field=metric_target_field)

    print("\n" + "="*50)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.2f}")
    print("="*50)

    # 6. Save File
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    final_output = {
        "checkpoint": args.checkpoint if args.checkpoint else "Base Model",
        "split": args.split,
        "metrics": metrics,
        "samples": detailed_results
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"\n✓ Đã lưu chi tiết kết quả tại: {args.output_file}")
    checkpoint_label = args.checkpoint if args.checkpoint else "Base Model"
    write_prediction_pairs(args.output_file, checkpoint_label, args.split, detailed_results)

if __name__ == "__main__":
    main()
