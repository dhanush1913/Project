import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralPEA(nn.Module):
    def __init__(self, input_dim, output_dim=6, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


class NeuralPEAWithAttention(nn.Module):
    def __init__(self, input_dim, output_dim=6, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.attention_query = nn.Parameter(torch.randn(output_dim, hidden_dim))
        nn.init.xavier_uniform_(self.attention_query)
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(output_dim)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for head in self.output_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    
    def compute_attention(self, h):
        h_exp = h.unsqueeze(1)  # (batch, 1, hidden_dim)
        q_exp = self.attention_query.unsqueeze(0)  # (1, num_maturities, hidden_dim)
        scores = h_exp * q_exp  # element-wise
        return F.softmax(scores, dim=-1)
    
    def forward(self, x, return_attention=False):
        h = self.encoder(x)
        attention = self.compute_attention(h)
        h_attended = attention * h.unsqueeze(1)
        
        outputs = torch.cat([
            self.output_heads[m](h_attended[:, m, :]) 
            for m in range(self.output_dim)
        ], dim=-1)
        
        return (outputs, attention) if return_attention else outputs
    
    def get_attention_weights(self, x):
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(x, return_attention=True)
            return attn.mean(dim=0)  # (num_maturities, hidden_dim)

class YieldDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features.values, dtype=torch.float32)
        self.Y = torch.tensor(targets.values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def create_model(input_dim, output_dim=6, hidden_dim=64, device='cpu'):
    model = NeuralPEA(input_dim, output_dim, hidden_dim)
    return model.to(device)
