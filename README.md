# CS898BD Deep Learning — Assignment 2

## **Q1: Tiny ImageNet dataset preparation** 
## **Q2: CIFAR-10 (ReLU vs Tanh) experiments**

## Repository Structure

```
.
├─ Q1/
│  ├─ CS898BD_ASSIGNMENT2_Q1.ipynb        # Builds the 100-class Tiny ImageNet subset
│  └─ tiny_100_alexnet/            # Created by the notebook: train/ val/ test/ + CSVs + meta.json
├─ Q2/
│  ├─ CS898BD_ASSIGNMENT2_Q2(1).ipynb        # CIFAR-10 experiments (ReLU vs Tanh)
│  └─ q2_runs/                     # Created by the notebook: relu/log.csv, tanh/log.csv, plots
├─ report/
│  └─ CS898BD_ASSIGNMENT2_Q2_REPORT.docx                # Final report for Question 2
└─ README.md
```

## Environment

You may run locally (conda) or in Google Colab.

### Option A — Local (conda; CPU is sufficient)

```bash
conda create -n dl25 python=3.10 -y
conda activate dl25
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install matplotlib pandas pillow ipykernel
python -m ipykernel install --user --name dl25 --display-name "Python (dl25)"
```

In Jupyter, select the kernel for q2 **Python (dl25)**.

### Option B — Google Colab

Open the notebook and install dependencies in the first cell:

```python
!pip -q install torch torchvision matplotlib pandas
```

(Optional) Runtime → Change runtime type → GPU. CPU also works.

---

## Q1 — Tiny ImageNet to 100-Class Subset (30k/10k/10k)

**Purpose.** Create a dataset compatible with AlexNet-style preprocessing and a fixed split.

**Steps.**

1. Download Tiny ImageNet (200 classes) from Stanford CS231n:
   `http://cs231n.stanford.edu/tiny-imagenet-200.zip`
2. Place the zip under `Q1/` (or update `ZIP_PATH` in the notebook).
3. Run `CS898BD_ASSIGNMENT2_Q1.ipynb` top to bottom.

**Outputs (created in `Q1/tiny_100_alexnet/`).**

* `train/`, `val/`, `test/` in ImageFolder layout.
* `train.csv`, `val.csv`, `test.csv`, and `meta.json` with split metadata.

**Checks.**

* Totals: train = 30,000; val = 10,000; test = 10,000.
* No overlap across splits (verification cells included).

**Citation for data source.** Tiny ImageNet (200 classes), Stanford CS231n: `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

---

## Q2 — CIFAR-10: Activation Function Experiments (ReLU vs Tanh)

**Objective.** Train the same 4-layer CNN twice on CIFAR-10, changing only the hidden-layer activation (ReLU vs Tanh). Stop training when **training error ≤ 25%**. Record **time per epoch** and produce the required plots.

**Procedure.**

1. Open `CS898BD_ASSIGNMENT2_Q2(1).ipynb` (Colab or local).
2. Run all cells. CIFAR-10 is downloaded automatically to `Q2/cifar10_data/`.
3. The notebook:

   * Trains the identical CNN with **ReLU** and **Tanh** in separate runs.
   * Applies early stopping at training error ≤ 0.25.
   * Logs per-epoch metrics to:

     * `Q2/q2_runs/relu/log.csv`
     * `Q2/q2_runs/tanh/log.csv`
   * Saves a combined figure to `Q2/q2_runs/q2_combined_error_and_time.png` containing:

     * Training error vs epochs (ReLU and Tanh)
     * Time per epoch (seconds) vs epochs (ReLU and Tanh)

**Final hyperparameters used (submitted runs).**

* ReLU learning rate: **0.01**; early stop at **epoch 5**.
* Tanh learning rate: **0.00019**; early stop at **epoch 30** (approximately **6×** ReLU epochs).

---

## Report

The Question 2 report is provided at `CS898BD_ASSIGNMENT2_Q2_REPORT.docx`.
It documents:

* Problem setup and model description.
* Training configuration and stopping rule.
* Original results (stop epochs and final metrics).
* Combined figure (training error vs epochs and time per epoch vs epochs).
* Explanation of the observed difference between ReLU and Tanh.

---

## Reproducibility Summary

* **Dataset.** CIFAR-10; normalization mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616).
* **Model.** 4-layer CNN: two 3×3 convolutional blocks (64/128 channels) with max pooling; FC-256 → logits-10. No BatchNorm or Dropout.
* **Initialization.** Kaiming/He for all weights; biases set to 0.
* **Optimizer.** SGD, momentum 0.9, batch size 128, weight decay 0.0 (for the comparison).
* **Early stopping.** Stop when training error ≤ 0.25.
* **Seed.** 42.

---

## Sources

* Tiny ImageNet (200 classes), Stanford CS231n: [http://cs231n.stanford.edu/tiny-imagenet-200.zip](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
* CIFAR-10: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* Krizhevsky, Sutskever, Hinton. “ImageNet Classification with Deep Convolutional Neural Networks.” NeurIPS 2012.

---

## License

This repository contains academic coursework. Datasets are subject to their original licenses and terms of use.
