import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

class DropoutNet(nn.Module):
    def __init__(self, input_size, activation="relu"):
        super(DropoutNet, self).__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "leaky_relu":
            act = nn.LeakyReLU()
        elif activation == "elu":
            act = nn.ELU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            act = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            act,
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            act,
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_neural(X_train, y_train, X_val, y_val, epochs=100, patience=10):
    print("Εκπαίδευση Neural Network")
    torch.manual_seed(42)

    results = {}

    for activation in ["relu", "leaky_relu", "tanh"]:
        print(f"\nActivation: {activation}")

        # Μετατροπή σε tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_val_t   = torch.FloatTensor(X_val)
        y_val_t   = torch.FloatTensor(y_val.values).unsqueeze(1)

        # DataLoader για batches
        dataset = TensorDataset(X_train_t, y_train_t)
        loader  = DataLoader(dataset, batch_size=64, shuffle=True)

        # Μοντέλο, loss, optimizer
        model     = DropoutNet(input_size=X_train.shape[1], activation=activation)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Early stopping
        best_val_loss    = float("inf")
        best_weights     = None
        patience_counter = 0
        train_losses     = []
        val_losses       = []

        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t)
                val_loss  = criterion(val_preds, y_val_t).item()
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_weights     = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping στο epoch {epoch+1}")
                    break

        # Επαναφορά καλύτερων weights
        model.load_state_dict(best_weights)
        print(f"Καλύτερο Val Loss: {best_val_loss:.4f}")
        results[activation] = {
            "model": model,
            "val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses
        }

    # Καλύτερο activation
    best_activation = min(results, key=lambda x: results[x]["val_loss"])
    print(f"\nΚαλύτερο activation function: {best_activation}")
    best_model        = results[best_activation]["model"]
    best_train_losses = results[best_activation]["train_losses"]
    best_val_losses   = results[best_activation]["val_losses"]

    # Αποθήκευση
    os.makedirs("models", exist_ok=True)
    torch.save(best_model.state_dict(), "models/neural_network.pt")
    print("Αποθηκεύτηκε στο models/neural_network.pt")

    return best_model, best_train_losses, best_val_losses