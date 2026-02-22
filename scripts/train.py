#!/usr/bin/env python3
"""
Train baseline CNN on Cats vs Dogs; log to MLflow; save model to models/baseline.pt.

Run from repo root:
  python scripts/train.py [--epochs 5] [--batch-size 32]

Reads data/processed/{train,val}.csv; writes models/baseline.pt and MLflow artifacts.
"""
import argparse
import os
import sys
from pathlib import Path

# Avoid matplotlib cache dir issues in CI / restricted envs
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.cnn import get_model
from src.model.dataset import CatsDogsDataset


def parse_args():
    p = argparse.ArgumentParser(description="Train Cats vs Dogs baseline CNN")
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--model-dir", type=Path, default=REPO_ROOT / "models", help="Directory to save model")
    p.add_argument("--train-csv", type=Path, default=REPO_ROOT / "data" / "processed" / "train.csv")
    p.add_argument("--val-csv", type=Path, default=REPO_ROOT / "data" / "processed" / "val.csv")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon (M1/M2/M3/M4)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_ds = CatsDogsDataset(args.train_csv, augment=True)
    val_ds = CatsDogsDataset(args.val_csv, augment=False)
    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=use_cuda
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    args.model_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(REPO_ROOT / "mlruns")
    mlflow.set_experiment("cats_vs_dogs")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        })
        train_losses, val_losses, val_accs = [], [], []

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            val_acc = accuracy_score(all_labels, all_preds)
            val_accs.append(val_acc)

            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}, step=epoch)
            print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Loss curve artifact
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train loss")
        plt.plot(val_losses, label="Val loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.title("Loss curves")
        loss_curve_path = REPO_ROOT / "loss_curve.png"
        plt.savefig(loss_curve_path, dpi=100)
        plt.close()
        mlflow.log_artifact(str(loss_curve_path))
        loss_curve_path.unlink(missing_ok=True)

        # Confusion matrix on validation set
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["cat", "dog"])
        ax.set_yticklabels(["cat", "dog"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.colorbar(im, ax=ax)
        plt.title("Validation confusion matrix")
        cm_path = REPO_ROOT / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=100)
        plt.close()
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink(missing_ok=True)

        # Save model (state_dict only for portability)
        model_path = args.model_dir / "baseline.pt"
        torch.save({"state_dict": model.state_dict(), "num_classes": 2}, model_path)
        mlflow.log_artifact(str(model_path))
        mlflow.pytorch.log_model(model, "model")

    print(f"Model saved to {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
