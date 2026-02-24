# HyperWalker

This repository contains the official implementation of **HyperWalker: Dynamic Hypergraph-Based Deep Diagnosis for Multi-Hop Clinical Modeling across EHR and X-Ray in Medical VLMs**.

## Overview

HyperWalker is a novel framework that reformulates clinical reasoning via dynamic hypergraphs and test-time training for medical vision-language models. It addresses the limitation of sample-isolated inference by incorporating longitudinal Electronic Health Records (EHRs) and structurally related patient examples into the diagnostic process.

### Key Features

- **Dynamic Hypergraph Construction (iBrochure)**: Models structural heterogeneity of EHR data and implicit high-order associations
- **Reinforcement Learning Agent (Walker)**: Navigates optimal diagnostic paths within the hypergraph
- **Linger Mechanism**: Multi-hop orthogonal retrieval strategy for comprehensive clinical coverage
- **Multi-modal Integration**: Combines EHR and X-Ray data for accurate diagnosis
- **State-of-the-Art Performance**: Achieves SOTA on MRG (MIMIC) and medical VQA (EHRXQA)

## Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0.0+
- CUDA 11.8+
- Ubuntu 22.04+

### Setup

```bash
git clone https://github.com/Bean-Young/HyperWalker.git
cd HyperWalker
pip install -r requirements.txt
```

## Project Structure

```
HyperWalker/
├── hyperbuild/          # Hypergraph construction modules
│   ├── data_loader.py
│   ├── embed_model.py
│   ├── build_hypergraph.py
│   └── hypergraph.py
├── embed_models/        # Multi-modal embedding models
│   ├── image_ehr_embed.py
│   ├── film_fusion.py
│   └── __init__.py
├── adatri/             # Adapter training modules
│   ├── train_adapter.py
│   ├── test_adapter.py
│   ├── data_loader.py
│   ├── compare_epochs.py
│   └── compare_epochs_simple.py
├── modeltr/            # Model training
│   └── train_model_f.py
├── metrics/            # Evaluation metrics
│   ├── ce_metrics.py
│   ├── nlg_metrics.py
│   └── CheXbert/       # CheXbert-based metrics
├── rl_finder/          # RL path finder
│   └── rl_path_finder.py
├── train.py            # Main training script
├── test.py             # Main testing script
└── find_ehr_by_study_id.py  # EHR retrieval utility
```

## Data Preparation

### Datasets

- **MIMIC-CXR**: Chest X-ray images with associated reports
- **EHRXQA**: Medical visual question answering dataset with EHR integration

### EHR Data

Use `find_ehr_by_study_id.py` to retrieve corresponding EHR data for each study:

```python
python find_ehr_by_study_id.py --study_id <STUDY_ID> --output_path <OUTPUT_PATH>
```

## Training

### Step 1: Build Hypergraph

Construct the dynamic hypergraph from EHR data:

```bash
python hyperbuild/build_hypergraph.py \
    --data_path <EHR_DATA_PATH> \
    --output_path <HYPERGRAPH_OUTPUT>
```

### Step 2: Train Adapter

Train the adapter with hypergraph guidance:

```bash
python adatri/train_adapter.py \
    --config configs/train_config.yaml \
    --hypergraph_path <HYPERGRAPH_OUTPUT> \
    --output_dir ./results/adapter
```

### Step 3: Full Model Training

Train the complete model:

```bash
python train.py \
    --config configs/full_train.yaml \
    --adapter_path ./results/adapter \
    --output_dir ./results/model
```

## Evaluation

### Medical Report Generation (MRG)

```bash
python test.py \
    --task mrg \
    --model_path <MODEL_CHECKPOINT> \
    --data_path <TEST_DATA> \
    --output_dir ./results/eval/mrg
```

### Medical Visual Question Answering (VQA)

```bash
python test.py \
    --task vqa \
    --model_path <MODEL_CHECKPOINT> \
    --data_path <TEST_DATA> \
    --output_dir ./results/eval/vqa
```

### Metrics

The framework supports multiple evaluation metrics:

- **NLG Metrics**: BLEU, ROUGE, METEOR for report generation
- **Clinical Metrics**: CheXbert-based label accuracy
- **VQA Metrics**: Accuracy, F1-score for question answering

## Results

Our method achieves state-of-the-art performance on:

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| MRG | MIMIC-CXR | BLEU-4 | XX.X |
| MRG | MIMIC-CXR | ROUGE-L | XX.X |
| VQA | EHRXQA | Accuracy | XX.X% |

*Full results will be updated upon paper publication.*

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{yang2025hyperwalker,
  title={HyperWalker: Dynamic Hypergraph-Based Deep Diagnosis for Multi-Hop Clinical Modeling across EHR and X-Ray in Medical VLMs},
  author={Yang, Yuezhe and Wang, Hao and Peng, Yige and Kim, Jinman and Bi, Lei},
  journal={arXiv preprint arXiv:2601.13919},
  year={2025}
}
```

## Acknowledgments

This project was developed at the Institute of Translational Medicine, Shanghai Jiao Tong University, in collaboration with the University of Sydney.

## Contact

For questions or issues, please contact:
- Yuezhe Yang (yuezheyang@sjtu.edu.cn)
- Project Page: [https://github.com/Bean-Young/HyperWalker](https://github.com/Bean-Young/HyperWalker)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
