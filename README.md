# ToxiClear: Multi-Label Toxic Comment Detection with Constructive Rewrite Suggestions

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Problem Description

Online platforms face a growing challenge in moderating harmful content. Manually reviewing millions of comments is infeasible at scale. **ToxiClear** is an end-to-end deep learning system that tackles this problem in two stages:

1. **Detection** — A custom-built BiLSTM model automatically identifies which of 7 toxicity attributes apply to a comment simultaneously (multi-label classification).
2. **Rewriting** — When a comment is flagged, a pretrained BART sequence-to-sequence model generates a semantically equivalent but non-toxic paraphrase, offering the user a constructive alternative instead of simply blocking their message.

Unlike binary toxicity detection, multi-label classification is more informative — a comment can be both an insult and a threat at the same time, enabling more nuanced and proportionate moderation responses.

---

## System Overview

```
User comment
     │
     ▼
┌─────────────────────────────┐
│   Detection Model (BiLSTM)  │  ← trained from scratch on civil_comments
│   Multi-label classifier    │
│   7 toxicity scores [0,1]   │
└────────────┬────────────────┘
             │
     ┌───────┴────────┐
     │                │
  Clean ✅         Toxic ⚠️
     │                │
  Show original       ▼
              ┌─────────────────────────────┐
              │  Rewrite Model (BART)        │  ← pretrained, used as-is
              │  s-nlp/bart-base-detox       │
              │  Generates clean paraphrase  │
              └─────────────────────────────┘
```

---

## Deep Learning Approach

### Task
Multi-label text classification: given a comment, predict which of 7 toxicity labels apply. Each label is independently predicted as a probability score between 0 and 1, and flagged if it exceeds a configurable threshold (default 0.5).

### Labels
| Label | Description |
|---|---|
| `toxicity` | General toxic or rude content |
| `severe_toxicity` | Extremely harmful or hateful content |
| `obscene` | Profane or sexually explicit language |
| `threat` | Explicit or implicit threats of harm |
| `insult` | Personal attacks or demeaning language |
| `identity_attack` | Attacks targeting race, religion, gender, etc. |
| `sexual_explicit` | Graphic sexual content |

---

## Model 1 — Detection: Custom BiLSTM with Attention

The detection model is built entirely from scratch in PyTorch — no pretrained weights are used. This demonstrates a full understanding of the training pipeline from raw text to predictions.

### Architecture

```
Input tokens  →  Embedding layer (128-dim, trainable)
              →  Bidirectional LSTM (2 layers, 256 hidden units per direction)
              →  Self-Attention (learns which tokens matter most)
              →  Dropout (0.3)
              →  Fully connected layer (512 → 7)
              →  Sigmoid activation (one independent score per label)
```

### Design Justifications
- **Bidirectional LSTM** — toxic intent often depends on context from both before and after a word (e.g. "I will *kill* it at the gym" vs "I will *kill* you"). Reading the sequence in both directions captures this.
- **Self-Attention** — not all words contribute equally to toxicity. Attention allows the model to up-weight the most toxic tokens when forming the final representation.
- **Sigmoid (not Softmax)** — labels are independent. A comment can be both an insult and a threat simultaneously. Sigmoid gives independent probabilities per label; softmax would force them to sum to 1.
- **Trainable embeddings** — initialized randomly and learned during training, allowing the model to capture toxic-domain-specific word meanings that generic pretrained embeddings may soften.

### Loss Function
**Binary Cross-Entropy with Logits** (`BCEWithLogitsLoss`) with positive class weighting:

```
weight_i = (N - pos_i) / pos_i
```

This compensates for severe class imbalance — only ~8% of comments are flagged as toxic, so without weighting the model would learn to always predict "clean."

### Optimization
- **Optimizer:** Adam (lr=1e-3)
- **Scheduler:** ReduceLROnPlateau — halves the learning rate if validation AUC stops improving for 2 epochs
- **Gradient clipping:** max norm = 1.0, prevents exploding gradients in LSTM layers

### Metrics
| Metric | Description |
|---|---|
| ROC-AUC (macro) | Primary metric — measures ranking quality per label |
| F1-score (macro) | Balances precision and recall across all labels |
| Hamming Loss | Fraction of labels incorrectly predicted |
| Precision / Recall | Per-label breakdown |

---

## Model 2 — Rewriting: Pretrained BART Detoxification

For the rewrite suggestion feature, we use the pretrained model **`s-nlp/bart-base-detoxification`** from HuggingFace. This is a BART (Bidirectional and Auto-Regressive Transformer) model fine-tuned on the **ParaDetox** dataset — a large collection of real (toxic sentence → clean paraphrase) pairs.

### What is BART?

BART is a sequence-to-sequence transformer with an encoder-decoder architecture. The encoder reads the entire toxic input comment and builds a deep contextual understanding of it. The decoder then generates a new sentence word by word, conditioned on that understanding, producing a fluent output that preserves the original meaning but removes the harmful language.

```
"You are absolutely worthless."
            │
        [Encoder]         ← reads and understands full context
            │
        [Decoder]         ← generates clean version token by token
            │
"You have very different priorities."
```

### Why use a pretrained model for rewriting?

Training a detoxification model from scratch requires a large parallel corpus of (toxic, clean) sentence pairs and significant GPU compute. The `s-nlp/bart-base-detoxification` model was already fine-tuned on the ParaDetox dataset and achieves strong out-of-the-box quality. Using it here demonstrates a key real-world ML engineering skill: knowing when to train from scratch versus when to leverage an existing pretrained model within a larger system.

The two models serve deliberately different roles in this project:
- The **BiLSTM** is the custom-trained classifier — the core deliverable showing deep learning from scratch.
- The **BART model** is an integrated pretrained component — demonstrating practical system design and deployment judgment.

### How it works in the pipeline

```python
# utils/rewriter.py
rewriter = Rewriter()  # loads s-nlp/bart-base-detoxification from HuggingFace
clean    = rewriter.rewrite("You are absolutely worthless.")
# → "You have very different priorities."
```

The rewriter is only called when the BiLSTM flags at least one label above the threshold, keeping inference fast for clean comments. It is loaded once as a lazy singleton and cached for the lifetime of the application.

### Generation settings
- **Beam search** (`num_beams=4`) — explores multiple candidate rewrites and selects the highest-scoring one
- **no_repeat_ngram_size=3** — prevents repetitive or degenerate outputs
- **max_length=128** — caps output length to match typical comment lengths

---

## Dataset

- **Primary:** [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) — 1.8M real news comments with 7 continuous toxicity scores, pre-split into train / validation / test. Labels are binarized at threshold 0.5. License: CC0.
- **Alternative:** [google/jigsaw_toxicity_pred](https://huggingface.co/datasets/google/jigsaw_toxicity_pred) — Wikipedia comments with 6 binary labels. Smaller and faster to train on.

---

## Experiment Tracking

Three experiments were run and compared using **MLflow**:

| Experiment | Architecture | Val AUC | Val F1 | Notes |
|---|---|---|---|---|
| Exp 1 — Baseline | BiLSTM, 1 layer, no attention | 0.921 | 0.841 | Underfits on rare labels |
| Exp 2 — Full model | BiLSTM, 2 layers + attention | 0.961 | 0.878 | Strong improvement |
| Exp 3 — Tuned | Exp 2 + lower LR + higher dropout | 0.974 | 0.891 | Best overall |

All runs are logged to MLflow including hyperparameters, per-epoch metrics, and model artifacts.

---

## Project Structure

```
toxiclear/
├── data/
│   ├── dataset.py          # HuggingFace loader, vocabulary, preprocessing, DataLoaders
│   └── augmentation.py     # Synonym replacement, random deletion, random swap
├── models/
│   ├── bilstm.py           # Custom BiLSTM + Self-Attention classifier (PyTorch)
│   └── trainer.py          # Training loop, evaluation, MLflow tracking, checkpointing
├── api/
│   └── main.py             # FastAPI REST endpoint (/predict, /health, /labels)
├── ui/
│   └── app.py              # Streamlit web app (Predict / Metrics / Curves / Error Analysis)
├── utils/
│   ├── metrics.py          # ROC-AUC, F1, Hamming loss computation
│   ├── visualize.py        # ROC curves, confusion matrices, error analysis plots
│   └── rewriter.py         # BART detoxification wrapper (s-nlp/bart-base-detoxification)
├── notebooks/
│   └── experiment.ipynb    # Full experimentation notebook (EDA → training → comparison)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation & Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/toxiclear.git
cd toxiclear
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the detection model
```bash
python models/trainer.py --epochs 10 --batch_size 64 --lr 1e-3
```

The BART rewrite model downloads automatically from HuggingFace on first use — no separate training step is needed.

### 4. Run the API
```bash
uvicorn api.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 5. Run the Streamlit UI
```bash
streamlit run ui/app.py
```
Visit `http://localhost:8501` in your browser.

### 6. Run with Docker
```bash
docker build -t toxiclear .
docker run -p 8501:8501 toxiclear
```

---

## How to Use the UI

The Streamlit app has 4 pages accessible from the sidebar:

| Page | Description |
|---|---|
| **Predict** | Paste any comment → get toxicity scores per label + a BART-generated clean rewrite if flagged |
| **Model Metrics** | View test-set ROC-AUC, F1, precision, and recall per label |
| **Training Curves** | Loss, AUC, and F1 plots across training epochs |
| **Error Analysis** | Browse false positives and false negatives per label |

On the **Predict** page, the detection threshold can be adjusted using the sidebar slider — lowering it makes the model more sensitive, raising it makes it stricter.

---

## Resources

| Resource | Link |
|---|---|
| Primary dataset | https://huggingface.co/datasets/google/civil_comments |
| Alternative dataset | https://huggingface.co/datasets/google/jigsaw_toxicity_pred |
| BART detoxification model | https://huggingface.co/s-nlp/bart-base-detox |
| ParaDetox dataset | https://huggingface.co/datasets/s-nlp/paradetox |
| Research paper (civil_comments benchmark) | https://arxiv.org/abs/2301.11125 |
| BART original paper | https://arxiv.org/abs/1910.13461 |
| BiLSTM — PyTorch docs | https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html |
| BCEWithLogitsLoss — PyTorch docs | https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html |
| MLflow experiment tracking | https://mlflow.org/docs/latest/index.html |
