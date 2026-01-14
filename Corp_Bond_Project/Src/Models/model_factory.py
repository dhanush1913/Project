try:
    from .neural_pea import NeuralPEA, NeuralPEAWithAttention, create_model
    from .attention_layer import MaturityAttention, create_attention_model
except ImportError:
    from neural_pea import NeuralPEA, NeuralPEAWithAttention, create_model
    from attention_layer import MaturityAttention, create_attention_model

DEFAULT_CONFIG = {
    'hidden_dim': 64,
    'num_maturities': 6,  # [y1, y2, y5, y10, y20, y30]
}

def get_pea_model(input_dim, config=None, device='cpu'):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    return create_model(
        input_dim=input_dim,
        output_dim=cfg['num_maturities'],
        hidden_dim=cfg['hidden_dim'],
        device=device
    )

def get_attention_pea_model(input_dim, config=None, device='cpu'):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    model = NeuralPEAWithAttention(
        input_dim=input_dim,
        output_dim=cfg['num_maturities'],
        hidden_dim=cfg['hidden_dim']
    )
    return model.to(device)

def list_available_models():
    return {
        'neural_pea': 'Base PEA - 2 hidden layers, no recurrence',
        'attention_pea': 'PEA with maturity-aware attention (Step 6)'
    }
