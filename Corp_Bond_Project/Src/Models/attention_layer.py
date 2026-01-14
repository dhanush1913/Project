import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):
    def __init__(self, hidden_dim, num_maturities):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_maturities = num_maturities
        self.attention_weights = nn.Parameter(torch.randn(num_maturities, hidden_dim))
        nn.init.xavier_uniform_(self.attention_weights)
    
    def forward(self, h):
        scores = torch.einsum('bh,mh->bm', h, self.attention_weights)  # (batch, num_maturities)
        attention = F.softmax(scores, dim=-1)  # (batch, num_maturities)
        contexts = attention.unsqueeze(-1) * h.unsqueeze(1)  # (batch, num_maturities, hidden_dim)
        return contexts, attention


class MaturityAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_maturities=6):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_maturities = num_maturities
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.attention_query = nn.Parameter(torch.randn(num_maturities, hidden_dim))
        nn.init.xavier_uniform_(self.attention_query)
        
        self.temperature = nn.Parameter(torch.ones(1))
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_maturities)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for head in self.output_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    
    def compute_attention(self, h):
        h_expanded = h.unsqueeze(1)  # (batch, 1, hidden_dim)
        query_expanded = self.attention_query.unsqueeze(0)  # (1, num_maturities, hidden_dim)
        
        scores = h_expanded * query_expanded  # (batch, num_maturities, hidden_dim)
        attention = F.softmax(scores / self.temperature, dim=-1)
        return attention
    
    def forward(self, x, return_attention=False):
        h = self.encoder(x)  # (batch, hidden_dim)
        
        attention = self.compute_attention(h)  # (batch, num_maturities, hidden_dim)
        h_attended = attention * h.unsqueeze(1)  # (batch, num_maturities, hidden_dim)
        
        outputs = []
        for m in range(self.num_maturities):
            out = self.output_heads[m](h_attended[:, m, :])  # (batch, 1)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=-1)  # (batch, num_maturities)
        
        if return_attention:
            return outputs, attention
        return outputs
    
    def get_attention_weights(self, x):
        self.eval()
        with torch.no_grad():
            _, attention = self.forward(x, return_attention=True)
            return attention.mean(dim=0)  # (num_maturities, hidden_dim)


def create_attention_model(input_dim, output_dim=6, hidden_dim=64, device='cpu'):
    model = MaturityAttention(input_dim, hidden_dim, output_dim)
    return model.to(device)


__all__ = ['FeatureAttention', 'MaturityAttention', 'create_attention_model']
