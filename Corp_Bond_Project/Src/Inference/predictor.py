import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Src.Models.neural_pea import NeuralPEA, NeuralPEAWithAttention

MATURITIES = ['y1', 'y2', 'y5', 'y10', 'y20', 'y30']


class YieldCurvePredictor:    
    def __init__(self, model_path=None, model_type='attention', input_dim=15, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        
        if model_type == 'attention':
            self.model = NeuralPEAWithAttention(input_dim, output_dim=6, hidden_dim=64)
        else:
            self.model = NeuralPEA(input_dim, output_dim=6, hidden_dim=64)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.to(device).eval()
        self._warmup()
    
    def _warmup(self, n=10):
        x = torch.randn(1, self.input_dim, device=self.device)
        for _ in range(n):
            with torch.no_grad():
                self.model(x)
    
    def predict(self, features):
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        with torch.no_grad():
            yields = self.model(features.to(self.device))
        
        yields_np = yields.cpu().numpy().squeeze()
        return {mat: float(yields_np[i]) for i, mat in enumerate(MATURITIES)}
    
    def predict_with_attention(self, features):
        if not isinstance(self.model, NeuralPEAWithAttention):
            raise TypeError("Attention requires NeuralPEAWithAttention model")
        
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        with torch.no_grad():
            yields, attention = self.model(features.to(self.device), return_attention=True)
        
        return {
            'yields': {mat: float(yields[0, i]) for i, mat in enumerate(MATURITIES)},
            'attention': attention.cpu().numpy().squeeze()
        }


def create_predictor(model_path=None, model_type='attention'):
    return YieldCurvePredictor(model_path, model_type)


if __name__ == "__main__":
    predictor = create_predictor()
    sample_input = np.random.randn(15)  # 15 macro-financial features
    
    result = predictor.predict(sample_input)
    print("Predicted Yield Curve:")
    for mat, val in result.items():
        print(f"  {mat}: {val:.3f}%")
