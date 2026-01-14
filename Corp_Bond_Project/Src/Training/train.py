import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Src.Models.neural_pea import NeuralPEA, YieldDataset
from Src.Training.losses import get_loss_fn

class PEATrainer:
    def __init__(self, model, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = get_loss_fn('mse')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def _to_device(self, X, Y):
        return X.to(self.device), Y.to(self.device)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for X, Y in train_loader:
            X, Y = self._to_device(X, Y)
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(X), Y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(X)
        return total_loss / len(train_loader.dataset)
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        for X, Y in val_loader:
            X, Y = self._to_device(X, Y)
            total_loss += self.loss_fn(self.model(X), Y).item() * len(X)
        return total_loss / len(val_loader.dataset)
    
    def fit(self, train_loader, val_loader, epochs=100, patience=10, verbose=True):
        best_val, wait = float('inf'), 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
            if val_loss < best_val:
                best_val, wait = val_loss, 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    if verbose: print(f"Early stopping at epoch {epoch+1}")
                    break
        self.model.load_state_dict(self.best_state)
        return self.history
    
    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        return self.model(X_t.to(self.device)).cpu().numpy()
    
    @torch.no_grad()
    def per_maturity_mse(self, X, Y):
        preds = self.predict(X)
        Y_np = Y.values if hasattr(Y, 'values') else Y
        return np.mean((preds - Y_np) ** 2, axis=0)

def prepare_data(features_path, targets_path, train_ratio=0.8, batch_size=32):
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    
    feat_cols = [c for c in features.columns if c != 'date']
    tgt_cols = [c for c in targets.columns if c != 'date']
    X, Y = features[feat_cols], targets[tgt_cols]
    
    n = int(len(X) * train_ratio)
    train_ds = YieldDataset(X.iloc[:n], Y.iloc[:n])
    val_ds = YieldDataset(X.iloc[n:], Y.iloc[n:])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X, Y, feat_cols, tgt_cols

def run_training(data_dir=None, epochs=100, lr=1e-3, hidden_dim=64, verbose=True):
    base = Path(data_dir) if data_dir else Path(__file__).parent.parent.parent / "Data" / "Processed"
    train_loader, val_loader, X, Y, feat_cols, tgt_cols = prepare_data(
        base / "normalized_features.csv", base / "maturity_targets.csv"
    )
    
    model = NeuralPEA(input_dim=len(feat_cols), output_dim=len(tgt_cols), hidden_dim=hidden_dim)
    trainer = PEATrainer(model, lr=lr)
    
    if verbose:
        print(f"Training Neural PEA: {len(feat_cols)} features → {len(tgt_cols)} maturities")
        print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    trainer.fit(train_loader, val_loader, epochs=epochs, verbose=verbose)
    return trainer, X, Y, tgt_cols

if __name__ == "__main__":
    trainer, X, Y, tgt_cols = run_training(epochs=100, verbose=True)
    print(f"\n✓ Training complete. Per-maturity MSE:")
    n = int(len(X) * 0.8)
    mse_per_mat = trainer.per_maturity_mse(X.iloc[n:], Y.iloc[n:])
    for mat, mse in zip(tgt_cols, mse_per_mat):
        print(f"  {mat}: {mse:.6f}")
