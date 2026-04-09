"""
trainer.py
----------
Full training loop for ToxicBiLSTM including:
  - Train / validation / test evaluation
  - Weighted BCEWithLogitsLoss for class imbalance
  - Adam optimizer with learning rate scheduling
  - Saving best model checkpoint
  - Training curve export
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss

from data.dataset import get_dataloaders, compute_label_weights, LABELS
from models.bilstm import ToxicBiLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion):
    """Run one evaluation pass. Returns loss, ROC-AUC, F1, hamming loss."""
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.cpu().numpy())

    probs  = np.concatenate(all_probs,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds  = (probs >= 0.5).astype(int)

    auc_scores = []
    for i in range(labels.shape[1]):
        if len(np.unique(labels[:, i])) > 1:  # skip if only 0s or only 1s
            auc_scores.append(roc_auc_score(labels[:, i], probs[:, i]))
    auc = np.mean(auc_scores) if auc_scores else 0.0

    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    hl  = hamming_loss(labels, preds)

    return total_loss / len(loader), auc, f1, hl


# ── Training loop ──────────────────────────────────────────────────────────────
def train(
    epochs:     int   = 5,
    batch_size: int   = 64,
    lr:         float = 1e-3,
    embed_dim:  int   = 128,
    hidden_dim: int   = 128,
    num_layers: int   = 1,
    dropout:    float = 0.5,
    subset:     int   = 200_000,
    save_dir:   str   = "checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, vocab = get_dataloaders(
        batch_size=batch_size, subset=subset
    )
    pos_weights = compute_label_weights(train_loader).to(DEVICE)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ToxicBiLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_labels=len(LABELS),
        dropout=dropout,
    ).to(DEVICE)

    print(f"Model parameters: {model.count_parameters():,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr , weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    # ── Tracking ──────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    best_auc = 0.0
    patience = 3
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_auc, val_f1, val_hl = evaluate(model, val_loader, criterion)
        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Hamming: {val_hl:.4f}"
        )

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {"model_state": model.state_dict(), "vocab": vocab},
                os.path.join(save_dir, "best_model.pt")
            )
            print(f"  ✓ New best AUC: {best_auc:.4f} — model saved.")
        
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Final test evaluation ──────────────────────────────────────────────────
    checkpoint = torch.load(os.path.join(save_dir, "best_model.pt"))
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_auc, test_f1, test_hl = evaluate(model, test_loader, criterion)

    print(f"\nTest AUC: {test_auc:.4f} | Test F1: {test_f1:.4f} | Hamming: {test_hl:.4f}")

    # ── Save training curves ───────────────────────────────────────────────────
    plot_training_curves(history, save_dir)
    return model, vocab


# ── Plot training curves ───────────────────────────────────────────────────────
def plot_training_curves(history: dict, save_dir: str):
    """Save loss and metric curves to disk."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#E74C3C")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#2471A3")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["val_auc"], label="Val ROC-AUC", color="#27AE60")
    axes[1].set_title("ROC-AUC (validation)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["val_f1"], label="Val F1", color="#8E44AD")
    axes[2].set_title("F1 Score (validation)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved to {path}")


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ToxicBiLSTM")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--subset",     type=int,   default=50_000)
    parser.add_argument("--save_dir",   type=str,   default="checkpoints")
    args = parser.parse_args()
    train(**vars(args))
