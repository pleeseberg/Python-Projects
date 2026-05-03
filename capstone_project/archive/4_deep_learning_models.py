
# script 4: deep learning models for sepsis prediction (lstm, gru, cnn1d)

# ============================================================
# 0. reproducibility
# ============================================================
import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# 1. output setup
# ============================================================
OUTPUT_DIR = "./outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"using device: {DEVICE}")
logging.info(f"output directory: {OUTPUT_DIR}")

# ============================================================
# 2. dataset class
# ============================================================
class SepsisDataset(Dataset):
    """torch dataset wrapper for sepsis tensors."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# 3. model definitions
# ============================================================
class BaseRNN(nn.Module):
    """shared architecture for lstm and gru."""
    def __init__(self, input_dim, hidden_dim=64, layers=1, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, layers, batch_first=True)
        else:
            raise ValueError("invalid rnn_type")
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # last timestep
        logits = self.fc(out)
        return torch.sigmoid(logits).squeeze(), logits


class CNN1D(nn.Module):
    """1d cnn. limited by single-timestep input."""
    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, timesteps)
        out = self.conv(x)
        out = out.mean(dim=2)
        logits = self.fc(out)
        return torch.sigmoid(logits).squeeze(), logits

# ============================================================
# 4. training with early stopping + scheduler
# ============================================================
def train_model(model, train_loader, val_loader, name, pos_weight):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

    best_val_loss = np.inf
    patience = 5
    trigger = 0
    history = []

    for epoch in range(1, 51):
        # ---------- train ----------
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            prob, logits = model(xb)
            loss = criterion(logits.squeeze(), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        # ---------- validation ----------
        model.eval()
        val_losses = []
        y_val_true, y_val_prob = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                prob, logits = model(xb)
                val_losses.append(criterion(logits.squeeze(), yb).item())
                y_val_true.extend(yb.cpu().numpy())
                y_val_prob.extend(prob.cpu().numpy())
        val_loss = np.mean(val_losses)

        # validation metrics
        val_auroc = roc_auc_score(y_val_true, y_val_prob)
        val_auprc = average_precision_score(y_val_true, y_val_prob)
        val_f1 = f1_score(y_val_true, (np.array(y_val_prob)>=0.5).astype(int))
        history.append([epoch, train_loss, val_loss, val_auroc, val_auprc, val_f1])
        logging.info(f"{name} | epoch {epoch} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | auroc: {val_auroc:.4f} | auprc: {val_auprc:.4f} | f1: {val_f1:.4f}")

        scheduler.step(val_loss)

        # ---------- early stopping ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": model.fc.in_features,
                "seed": SEED,
                "timestamp": time.time()
            }, f"{OUTPUT_DIR}/{name}_best.pt")
        else:
            trigger += 1
            if trigger >= patience:
                logging.info(f"{name} early stopping triggered at epoch {epoch}")
                break

    # save training history
    pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "val_f1"]).to_csv(
        f"{OUTPUT_DIR}/{name}_loss_history.csv", index=False
    )

# ============================================================
# 5. evaluation
# ============================================================
def evaluate_model(model, loader):
    model.eval()
    preds, labels, logits_all = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            prob, logits = model(xb)
            preds.extend(prob.cpu().numpy())
            logits_all.extend(logits.squeeze().cpu().numpy())
            labels.extend(yb.numpy())
    return np.array(labels), np.array(preds), np.array(logits_all)

# ============================================================
# 6. main pipeline
# ============================================================
def main():
    # ---------- load data ----------
    try:
        X_train = np.load("../outputs/X_train.npy")
        X_test  = np.load("../outputs/X_test.npy")
        y_train = np.load("../outputs/y_train.npy")
        y_test  = np.load("../outputs/y_test.npy")
    except Exception as e:
        logging.error(f"data loading failed: {e}")
        return

    assert X_train.shape[1] == 34, "feature mismatch! expected 34 features."

    # ---------- reshape for timesteps ----------
    X_train = X_train[:, np.newaxis, :]
    X_test  = X_test[:, np.newaxis, :]
    logging.info(f"input shape: (timesteps={X_train.shape[1]}, features={X_train.shape[2]})")

    # ---------- class imbalance ----------
    pos_rate = y_train.mean()
    logging.info(f"training sepsis prevalence: {pos_rate:.4%}")
    pos_weight = torch.tensor((1 - pos_rate) / pos_rate).to(DEVICE)

    # ---------- dataset & dataloader ----------
    dataset = SepsisDataset(X_train, y_train)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)
    test_loader  = DataLoader(SepsisDataset(X_test, y_test), batch_size=64)

    input_dim = X_train.shape[2]

    # ---------- define models ----------
    models = {
        "LSTM": BaseRNN(input_dim, rnn_type="lstm").to(DEVICE),
        "GRU": BaseRNN(input_dim, rnn_type="gru").to(DEVICE),
        "CNN1D": CNN1D(input_dim).to(DEVICE)
    }

    results = []

    for name, model in models.items():
        logging.info(f"training {name}")
        train_model(model, train_loader, val_loader, name, pos_weight)

        # load best checkpoint
        checkpoint = torch.load(f"{OUTPUT_DIR}/{name}_best.pt")
        model.load_state_dict(checkpoint["model_state"])

        y_true, y_prob, y_logits = evaluate_model(model, test_loader)
        y_pred = (y_prob >= 0.5).astype(int)

        # save predictions + logits
        pd.DataFrame({
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "logits": y_logits
        }).to_csv(f"{OUTPUT_DIR}/{name}_test_predictions.csv", index=False)

        # compute metrics
        metrics = {
            "model": name,
            "auroc": roc_auc_score(y_true, y_prob),
            "auprc": average_precision_score(y_true, y_prob),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }

        # save confusion matrix
        pd.DataFrame(confusion_matrix(y_true, y_pred)).to_csv(
            f"{OUTPUT_DIR}/{name}_confusion_matrix.csv", index=False
        )

        results.append(metrics)

        # save config for reproducibility
        config = {
            "input_dim": input_dim,
            "rnn_type": getattr(model, "rnn_type", "CNN1D"),
            "hidden_dim": model.rnn.hidden_size if hasattr(model, "rnn") else None,
            "layers": model.rnn.num_layers if hasattr(model, "rnn") else None,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "seed": SEED
        }
        pd.DataFrame([config]).to_csv(f"{OUTPUT_DIR}/{name}_config.csv", index=False)

        # ---------- save lstm attention weights ----------
        if name == "LSTM":
            lstm_attentions_list = []
            lstm_model = model
            lstm_model.eval()
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(DEVICE)
                    prob, logits = lstm_model(xb)
                    if hasattr(lstm_model, "attention_weights"):
                        lstm_attentions_list.append(lstm_model.attention_weights.cpu().numpy())
                    else:
                        logging.warning("lstm model has no 'attention_weights' attribute. skipping attention save.")
                        break
            if lstm_attentions_list:
                lstm_attentions = np.vstack(lstm_attentions_list)
                np.save(os.path.join(OUTPUT_DIR, "lstm_att_attentions.npy"), lstm_attentions)
                logging.info(f"lstm attention weights saved → {OUTPUT_DIR}/lstm_att_attentions.npy")

    # save overall metrics
    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/deep_learning_metrics.csv", index=False)
    logging.info(f"deep learning results saved → {OUTPUT_DIR}/deep_learning_metrics.csv")


if __name__ == "__main__":
    main()
