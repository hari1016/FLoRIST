# FLoRIST: Singular Value Thresholding for Efficient and Accurate Federated Fine-Tuning of Large Language Models

Official implementation of **FLoRIST**, a federated fine-tuning framework that performs low-rank aggregation in latent space using **Singular Value Thresholding (SVT)** to achieve a strong balance between accuracy, communication efficiency, and scalability for large language models.

📄 *Accepted to MLSys 2026*  
This repository accompanies the camera-ready version of the paper.

---

## Overview

Federated fine-tuning of LLMs faces a fundamental tension: large models require expressive updates, but communication constraints and client heterogeneity limit what can be transmitted. Existing LoRA-based methods either introduce aggregation noise, rely on expensive full-matrix decompositions, or scale poorly when client ranks differ.

FLoRIST resolves this by operating directly in the low-rank latent adapter space instead of constructing dense global updates. The server performs an efficient decomposition of stacked client adapters and applies singular value thresholding to retain only the most informative components. The result is a compact global adapter that preserves performance while dramatically reducing communication.

In practice, this design yields higher MMLU accuracy, large communication savings (often 5×–350×), lower server computation compared to full SVD baselines, and stable behavior under heterogeneous client ranks.

---

## Key Ideas

FLoRIST introduces a unified framework for federated LoRA aggregation built around three principles:

- **Noise-free aggregation** that avoids the cross-term artifacts present in FedIT-style averaging
- **Efficient SVD without forming the dense update matrix**, making server computation scalable
- **Energy-based rank selection** via singular value thresholding to remove redundant components

This leads to adaptive per-layer compression and a global adapter that captures the shared signal across clients while filtering noisy updates. Across TinyLlama and Llama-3.2-1B models, and across Dolly, Alpaca, and Wizard datasets, FLoRIST consistently matches or exceeds the accuracy of FedIT, FFA-LoRA, FLoRA, and FlexLoRA while offering substantially higher communication efficiency.

---

<p align="center">
  <img src="./figures/workflow-8.png" width="75%" alt="FLoRIST Workflow">
</p>

---

## Experimental Setup

All results follow the protocol used in the MLSys paper. Training uses 100 total clients with 10 sampled per communication round for 75 rounds under non-IID partitions. LoRA is applied to attention layers only, and performance is evaluated on a 1,444-sample subset of MMLU.

Two client configurations are studied:

**Homogeneous setting.**  
All clients use LoRA rank 16.

**Heterogeneous setting.**  
Client ranks follow a heavy-tail distribution reflecting realistic device imbalance:

- 40 clients use rank 4  
- 20 clients use rank 8  
- 20 clients use rank 16  
- 10 clients use rank 32  
- 10 clients use rank 64  

---

## Requirements

```bash
pip install -r requirements.txt
````

Tested with PyTorch 2.x, CUDA 12, and NVIDIA H100/A100 GPUs.

---

## Datasets

We follow the data format used in prior federated LoRA work. Each sample contains:

```
instruction
input (optional)
output
```

Datasets are downloaded and placed in:

Wizard → `./data_wiz/`
[https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k)

Alpaca → `./data_alpaca/`
[https://huggingface.co/datasets/tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

Dolly → `./data/`
[https://huggingface.co/datasets/databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

---

## Training

### Homogeneous configuration

```bash
python main.py \
  --global_model tinyllama \
  --data_path ./data \
  --num_clients 100 \
  --clients_per_round 10 \
  --num_communication_rounds 75 \
  --local_num_epochs 3 \
  --method florist \
  --threshold 0.90
```

### Heterogeneous configuration

```bash
python main.py \
  --global_model tinyllama \
  --data_path ./data_wiz \
  --num_clients 100 \
  --clients_per_round 10 \
  --num_communication_rounds 75 \
  --local_num_epochs 3 \
  --method florist \
  --threshold 0.85 \
  --heter True
```

For FedIT or FFA-LoRA baselines, add:

```
--zero_padding True
```

---

## Available Methods

The framework supports multiple aggregation strategies:

```
florist   (proposed method)
flora
fedit
flex
ffa
```

---

## Threshold Selection

The SVT threshold τ ∈ [0.80, 0.99] controls the retained rank of the global adapter. Lower thresholds produce stronger compression and higher communication efficiency, while higher thresholds prioritize maximal accuracy.

In experiments, τ is selected via binary search to match or exceed baseline performance. Thresholding acts as an implicit regularizer that filters noisy client-specific components and preserves shared structure.

---

## Evaluation

Each run automatically reports:

* MMLU accuracy
* convergence curves
* communication efficiency
* adapter rank statistics

Communication efficiency is defined as the inverse of total transmitted parameters. Lower effective rank corresponds to higher efficiency.

---

## Results Summary

Across all evaluated settings, FLoRIST achieves the strongest overall accuracy–efficiency trade-off. It frequently matches or exceeds the best baseline accuracy while delivering the highest communication efficiency and significantly lower server compute than full-matrix SVD methods.

Representative findings include:

* up to **349× less communication** than full fine-tuning
* up to **108× higher efficiency** than FLoRA
* ~7× lower server FLOPs than FlexLoRA
* adaptive rank compression that varies across layers
* higher intrinsic dimensionality in intermediate layers
* consistent redundancy in value projections vs query projections

---

## Communication Cost Example

TinyLlama on Wizard (homogeneous):

| Method      | Download (MB) |
| ----------- | ------------- |
| Full FT     | 2076          |
| FedIT       | 45            |
| FLoRA       | 45            |
| FFA         | 22            |
| **FLoRIST** | **8.4**       |

---

## License

Apache 2.0

---

## Contributing

We welcome reproducibility improvements, dataset integrations, and benchmarking extensions. Please open an issue before large architectural changes.

