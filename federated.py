import copy
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from skorch import NeuralNetClassifier

class VirtualIoTClient:
    """
    Simulates a virtual IoT edge device (fl_i) with decentralized private data.
    """
    def __init__(self, client_id: str, data_loader, device: torch.device):
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.sample_count = len(data_loader.dataset)

    def train_round(self, model_template: nn.Module, global_weights: dict, local_epochs: int = 1, lr: float = 0.001):
        """
        Executes local training round on device using received global weights.
        Returns: updated weights, local training duration
        """
        t0 = time.time()
        local_model = copy.deepcopy(model_template).to(self.device)
        local_model.set_weights(global_weights)
        local_model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)

        for _ in range(local_epochs):
            for x_batch, y_batch in self.data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = local_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        duration = time.time() - t0
        return local_model.get_weights(), duration, self.sample_count


class FLAverageServer:
    """
    Implements the central aggregation server and the flaverage() function
    as specified in Algorithm 1 of the paper.
    """
    def __init__(self, global_model: nn.Module, device: torch.device):
        self.global_model = global_model.to(device)
        self.device = device

    def aggregate(self, client_results: list):
        """
        Federated Averaging (FedAvg):
        M_w = sum_{k=1}^K (n_k / n) * m_w^k
        """
        t0 = time.time()
        total_samples = sum([res[2] for res in client_results])
        if total_samples == 0:
            return self.global_model.get_weights(), 0.0

        global_dict = copy.deepcopy(client_results[0][0])
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)

        for client_weights, _, sample_count in client_results:
            weight_factor = sample_count / total_samples
            for key in global_dict.keys():
                global_dict[key] += client_weights[key].float() * weight_factor

        for key in global_dict.keys():
            if not global_dict[key].is_floating_point():
                global_dict[key] = global_dict[key].to(client_results[0][0][key].dtype)

        self.global_model.set_weights(global_dict)
        agg_time = time.time() - t0
        return global_dict, agg_time


def evaluate_model(model: nn.Module, test_loader, device: torch.device):
    """
    Computes Accuracy, Precision, Recall, F1-Score as described in Section IV-B
    using the Skorch NeuralNetClassifier wrapper to interface PyTorch models
    with Scikit-Learn evaluation packages.
    """
    model.eval()
    
    # Collect test tensors
    X_test_list = []
    y_test_list = []
    for x_batch, y_batch in test_loader:
        X_test_list.append(x_batch.numpy() if isinstance(x_batch, torch.Tensor) else x_batch)
        y_test_list.append(y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch)

    X_test = np.concatenate(X_test_list, axis=0).astype(np.float32)
    y_true = np.concatenate(y_test_list, axis=0).astype(np.int64)

    # Wrap PyTorch model with Skorch NeuralNetClassifier for Scikit-Learn evaluation
    skorch_model = NeuralNetClassifier(
        module=copy.deepcopy(model).cpu(),
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        max_epochs=0,
        device="cpu",
        warm_start=True,
        verbose=0
    )
    skorch_model.initialize()
    skorch_model.module_.load_state_dict(model.state_dict())

    # Generate predictions via Skorch
    y_pred = skorch_model.predict(X_test)
    y_probs = skorch_model.predict_proba(X_test)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs
    }


def train_federated(model_template: nn.Module, client_loaders: list, test_loader, rounds: int = 10, local_epochs: int = 1, lr: float = 0.001, device: str = "cpu"):
    """
    Executes Algorithm 1: Anomaly Detection With Federated Learning.
    Returns: trained global model, history dict, total training time
    """
    dev = torch.device(device)
    server = FLAverageServer(copy.deepcopy(model_template), dev)
    clients = [VirtualIoTClient(f"fl_{i+1}", ldr, dev) for i, ldr in enumerate(client_loaders)]
    
    history = []
    total_training_time = 0.0

    current_weights = server.global_model.get_weights()

    for r in range(rounds):
        client_updates = []
        client_times = []

        # Local training on each virtual device fl_i
        for client in clients:
            weights, dur, s_count = client.train_round(model_template, current_weights, local_epochs=local_epochs, lr=lr)
            client_updates.append((weights, dur, s_count))
            client_times.append(dur)

        # flaverage aggregation at central server
        current_weights, agg_time = server.aggregate(client_updates)
        
        # Parallel simulation: training time is max client time + aggregation time + network communication latency
        round_duration = max(client_times) + agg_time
        total_training_time += round_duration

        metrics = evaluate_model(server.global_model, test_loader, dev)
        metrics["round"] = r + 1
        metrics["round_time"] = round_duration
        metrics["total_time"] = total_training_time
        history.append(metrics)

    return server.global_model, history, total_training_time


def train_centralized(model_template: nn.Module, central_train_loader, test_loader, epochs: int = 10, lr: float = 0.001, device: str = "cpu"):
    """
    Trains non-FL baseline model on centralized training data.
    """
    dev = torch.device(device)
    model = copy.deepcopy(model_template).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    total_training_time = 0.0

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        for x_batch, y_batch in central_train_loader:
            x_batch = x_batch.to(dev)
            y_batch = y_batch.to(dev)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        ep_duration = time.time() - t0
        total_training_time += ep_duration

        metrics = evaluate_model(model, test_loader, dev)
        metrics["epoch"] = ep + 1
        metrics["epoch_time"] = ep_duration
        metrics["total_time"] = total_training_time
        history.append(metrics)

    return model, history, total_training_time
