<p align="center">
  <img src="banner.png" width="900"/>
</p>

# Deep Learning for Dog Breed Image Classification

A complete deep-learning pipeline for **binary dog breed classification** (Golden Retriever vs. Collie) and representation learning. The project includes a custom CNN, a transfer-learning workflow, and a Vision Transformer implemented from scratch. It is intended for machine-learning students, researchers, and engineers who want to see an end-to-end computer-vision system in a single, readable codebase.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Quickstart](#quickstart)
3. [Key Features](#key-features)
4. [Architecture](#architecture)
5. [Dependencies and Installation](#dependencies-and-installation)
6. [Usage Examples](#usage-examples)
7. [Vision Transformer Walkthrough](#vision-transformer-walkthrough)
8. [Challenge Model & Prediction Pipeline](#challenge-model--prediction-pipeline)
9. [FAQ](#faq)
10. [Contributing](#contributing)
11. [Acknowledgements](#acknowledgements)

---

## Project Overview

This repository implements a full deep-learning workflow for image classification of dog breeds, focusing on five deliverables:

- Binary classification of Golden Retriever vs. Collie
- Supervised pretraining on ten additional breeds
- Transfer-learning experiments from the pretraining step to the target task
- A fully custom Vision Transformer (attention, patch embedding, encoder blocks, `[CLS]` pooling)
- A configurable challenge model for hidden-test leaderboard evaluation

The code was developed for an advanced machine-learning course at the University of Michigan to explore **representation learning, model generalization, and the architectural tradeoffs** between convolutional networks and transformers.

---

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/oaydas/dog-breed-classification.git
cd dog-breed-classification
```

### 2. Set up the environment

Create a Conda environment and install all dependencies:

```bash
conda create --name dl-dogs python=3.9
conda activate dl-dogs
pip install -r requirements.txt
```

> **Note:** the Conda environment provides an isolated Python installation, and the `pip install` command installs the required packages into it. Both steps are necessary — the Conda step alone will not pull in the PyTorch and scientific-Python dependencies this repo relies on.

### 3. Train the baseline CNN
```bash
python train_cnn.py
```

### 4. Evaluate the trained model
```bash
python test_cnn.py
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| CNN Baseline | Custom convolutional network for the binary classification target |
| Transfer Learning | Pretrain on 10 dog breeds, then fine-tune on Golden vs. Collie |
| Vision Transformer | Full ViT from scratch — attention, patch embedding, encoder blocks |
| Challenge Model | Modular architecture for inference on a held-out hidden test set |
| Early Stopping | Automatic checkpointing with validation-based stopping |
| AUROC Tracking | Robust evaluation metric under class imbalance |

---

## Architecture

<p align="center">
  <img src="architecture.png" width="850"/>
</p>

<p align="center"><em>Figure 1. Dog Breed Classification — End-to-End Pipeline. Preprocessing is its own labeled stage (<code>dataset.py</code>); the CNN and Vision Transformer are distinct feature extractors rather than a single block; every arrow is labeled with the data type it carries.</em></p>

The pipeline has five stages. The descriptions below explicitly name the source file that implements each stage, so that a reader browsing the repository can jump directly from the diagram into the code.

1. **Input stage (*training / test images*).** Labeled images of Golden Retrievers and Collies are used during training; unlabeled test images follow the same pipeline during inference.
2. **Data preprocessing (`dataset.py`).** Resize, per-channel zero-mean unit-variance normalization, and augmentation (random flips and crops at train time only). The same tensor representation is fed to both backbones downstream.
3. **Feature extractor (`models/cnn.py` or `models/vit.py`).** Either a stacked convolutional backbone with ReLU and max-pooling, or a Vision Transformer that splits each image into 16×16 patches, adds learned positional encodings, and applies multi-head self-attention through a stack of encoder blocks. Both emit a fixed-length feature vector.
4. **Classifier head.** A fully-connected layer maps the feature vector to a two-class probability through softmax.
5. **Model training & evaluation (`utils.py`).** Per-epoch loss and accuracy metrics, AUROC tracking on the validation split, and early stopping on validation AUROC with automatic best-checkpoint saving.

---

## Dependencies and Installation

Required packages:

- Python 3.9+
- PyTorch
- NumPy, Pandas, scikit-learn, matplotlib, tqdm

Install the full dependency set with:

```bash
pip install -r requirements.txt
```

---

## Usage Examples

### CNN training
```bash
python train_cnn.py
```

Representative output:
```
Epoch [1/50] | Train Loss: 0.682 | Val Acc: 0.621 | Val AUROC: 0.658
Epoch [2/50] | Train Loss: 0.534 | Val Acc: 0.743 | Val AUROC: 0.791
...
Early stopping at epoch 32. Best Val AUROC: 0.934
Model saved to checkpoints/cnn_best.pt
```

### Transfer learning
```bash
python train_source.py   # pretrain on 10 breeds
python train_target.py   # fine-tune on Golden vs. Collie
```

### Vision Transformer
```bash
python train_vit.py
```

### Generate challenge predictions
```bash
python predict_challenge.py submission
```

Produces a `submission.csv` file formatted for leaderboard evaluation.

---

## Vision Transformer Walkthrough

<p align="center">
  <img src="vit.gif" height="300"/>
</p>

The ViT implementation in `models/vit.py` includes:

- Patch embedding (16×16)
- Multi-Head Self-Attention
- LayerNorm + MLP blocks
- `[CLS]` token pooling
- Positional encodings

It is written manually without using PyTorch's high-level transformer API, in order to make every step of the attention mechanism explicit and readable. The accompanying animation (`vit.gif`), produced with Manim, visualizes patch extraction, embedding, and attention flow.

---

## Challenge Model & Prediction Pipeline

The challenge system supports:

- Hidden test labels
- CSV submission formatting
- Ensemble-ready outputs
- AUROC-based validation selection

Running `predict_challenge.py` loads the best checkpoint, performs inference on the challenge test set, applies softmax to produce class probabilities, and writes results to a CSV file.

---

## FAQ

**Q: Why use a ViT on such small images?**
A: To compare the inductive biases of the two architectures directly. Convolutional networks enforce translation-equivariance and locality; transformers do not, and must learn those biases from data. On a small dataset this asymmetry is exactly what makes the comparison interesting.

**Q: Is a GPU required?**
A: No. A single CPU run completes in roughly twenty minutes per model. A GPU shortens the loop but is not a dependency.

**Q: Can I extend this to multi-class classification?**
A: Yes — change the final layer and loss function in `models/target.py`. The data loaders already support multiple classes.

**Q: Why AUROC instead of accuracy?**
A: AUROC is robust under class imbalance, which can appear during transfer learning, and it is a probabilistic metric rather than a thresholded one. The early-stopping logic uses AUROC for the same reason.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add experiments or visualizations
4. Submit a pull request

---

## Acknowledgements

- PyTorch Team
- Stanford CS231n
- Dosovitskiy et al., *An Image is Worth 16×16 Words*
- University of Michigan EECS 445
- ViT animation created with Manim (3Blue1Brown)
