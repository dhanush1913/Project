import torch
import torch.nn as nn

class MaturityMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, y_pred, y_true):
        mse = (y_pred - y_true) ** 2
        return mse.mean() if self.reduction == 'mean' else mse.sum()

class WeightedMaturityLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights  # e.g., [1, 1, 1.5, 2, 2.5, 3] for [1Y, 2Y, 5Y, 10Y, 20Y, 30Y]
    
    def forward(self, y_pred, y_true):
        mse = (y_pred - y_true) ** 2
        if self.weights is not None:
            w = torch.tensor(self.weights, device=y_pred.device, dtype=y_pred.dtype)
            mse = mse * w
        return mse.mean()

def get_loss_fn(loss_type='mse', **kwargs):
    if loss_type == 'mse':
        return MaturityMSELoss()
    elif loss_type == 'weighted':
        return WeightedMaturityLoss(**kwargs)
    return nn.MSELoss()
