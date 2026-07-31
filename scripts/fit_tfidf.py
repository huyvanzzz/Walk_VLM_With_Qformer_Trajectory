import argparse
import yaml
import sys
import json
sys.path.append('.')

from scripts.metrics import VLMMetrics
from preprocessing import format_ground_truth, get_response_format, map_metadata_to_ground_truth
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Fit TF-IDF vectorizer on training data")
    parser.add_argument('--config', required=True, help='Path to config.yaml')
    parser.add_argument('--output', default='tfidf_vectorizer.pkl', help='Output pickle file')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    response_format = get_response_format(config)
    
    # Load training dataset
    print("Loading training dataset...")
    metadata = load_dataset(
        config['data']['name'],
        data_files={"train": "train.json"},
        split="train"
    )
    
    print(f"Total samples: {len(metadata)}")
    
    # Extract all instructions from training set
    print("Extracting instructions...")
    corpus = []
    
    for i, sample in enumerate(metadata):
        try:
            if response_format == "direct_text":
                instruction = format_ground_truth(sample, response_format)
            else:
                gt = map_metadata_to_ground_truth(sample)
                instruction = gt.instruction
            
            if instruction and len(instruction) > 0:
                corpus.append(instruction)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(metadata)} samples")
                
        except Exception as e:
            print(f"  Warning: Error at sample {i}: {e}")
            continue
    
    print(f"\n✓ Collected {len(corpus)} valid instructions")
    
    # Fit TF-IDF
    print("\nFitting TF-IDF vectorizer...")
    metrics = VLMMetrics(tfidf_path=args.output)
    metrics.fit_tfidf(corpus)
    
    print(f"\n{'='*60}")
    print("TF-IDF FITTING COMPLETED")
    print(f"{'='*60}")
    print(f"  Output file: {args.output}")
    print(f"  Corpus size: {len(corpus)} instructions")
    print(f"  Ready for evaluation!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
