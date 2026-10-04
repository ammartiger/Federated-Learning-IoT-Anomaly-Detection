import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class IoT_GRU(nn.Module):
    """
    GRU neural network model for IoT network anomaly detection
    as specified in Table II of the paper:
    'Federated-Learning-Based Anomaly Detection for IoT Security Attacks'
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0, num_classes: int = 5):
        super(IoT_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_classes = num_classes

        # In PyTorch, dropout in nn.GRU is only applied between layers if num_layers > 1
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )
        
        # Optional dropout layer after GRU before classification head
        self.fc_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch_size, window_size, input_size)
        Returns: logits of shape (batch_size, num_classes)
        """
        # GRU returns (output, h_n)
        # output shape: (batch_size, window_size, hidden_size)
        out, _ = self.gru(x)
        # Select representation at last time step
        last_step = out[:, -1, :]
        last_step = self.fc_dropout(last_step)
        logits = self.fc(last_step)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns class probabilities using Softmax.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def get_weights(self) -> dict:
        """
        Extract model state dictionary for federated aggregation.
        """
        return copy.deepcopy(self.state_dict())

    def set_weights(self, state_dict: dict):
        """
        Load model state dictionary from central server.
        """
        self.load_state_dict(copy.deepcopy(state_dict))


# Architectures specified in Table II of the paper
GRU_CONFIGS = {
    "GRU-1": {"num_layers": 2, "dropout": 0.01, "hidden_size": 256},
    "GRU-2": {"num_layers": 1, "dropout": 0.00, "hidden_size": 100},
    "GRU-3": {"num_layers": 1, "dropout": 0.00, "hidden_size": 200},
    "GRU-4": {"num_layers": 2, "dropout": 0.00, "hidden_size": 100},
}

def create_model(model_name: str, input_size: int, num_classes: int = 5) -> IoT_GRU:
    """
    Factory function to instantiate models GRU-1 through GRU-4.
    """
    if model_name not in GRU_CONFIGS:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from: {list(GRU_CONFIGS.keys())}")
    
    cfg = GRU_CONFIGS[model_name]
    return IoT_GRU(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_classes=num_classes
    )
