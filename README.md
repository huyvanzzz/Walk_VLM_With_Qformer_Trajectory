# NavVLM

Official implementation of **NavVLM: Integrating Object Tracking and Q-Former Compression for Assistive Navigation of Visually Impaired Users**.

NavVLM is a vision-language framework for generating safety-aware navigation instructions for visually impaired users.

The proposed framework combines:

- **Q-Former-based visual token compression** to reduce the visual sequence from 256 to 32 tokens.
- **Object tracking and trajectory encoding** to capture the motion of surrounding objects across consecutive frames.
- **Trajectory–visual token fusion** to integrate visual appearance and object-level motion information.

Experiments on the Walking Awareness Dataset show that NavVLM achieves ROUGE-1, ROUGE-2, and ROUGE-L scores of **0.473**, **0.322**, and **0.433**, respectively. Compared with the corresponding InternVL baseline, the framework reduces end-to-end latency by **4.37%** and improves decoding throughput by **10.95%**.

## Repository Overview

This repository contains the implementation for:

- Tracking-aware data preparation
- Tracking encoder pretraining
- Q-Former-based visual token compression
- CLS fusion and token concatenation
- Vision-language model fine-tuning
- Inference and evaluation

## Repository Structure

```text
Walk_VLM_With_Qformer_Trajectory/
├── gptscore/                       # GPTScore evaluation
├── model/                          # Model configurations and modules
├── tests/                          # Testing and debugging scripts
│
├── build_frame_index.py            # Builds frame indices for the WAD dataset
├── data.py                         # Data-loading utilities
├── preprocessing.py                # Input preprocessing utilities
├── wad_dataset.py                  # WAD dataset loader
│
├── pretrain_dataset.py             # Dataset for tracking encoder pretraining
├── train_pretrain.py               # Tracking encoder pretraining
├── pretrain_checkpoint_verify.py   # Pretrained checkpoint verification
│
├── trajectory_branch.py            # Tracking encoder and fusion modules
├── trajectory_trainability.py      # Controls trainable trajectory modules
├── qformer_bridge.py               # Q-Former integration and token compression
│
├── train.py                        # Main fine-tuning script
├── optimizer_state_utils.py        # Optimizer checkpoint utilities
├── logutil.py                      # Logging utilities
└── README.md
