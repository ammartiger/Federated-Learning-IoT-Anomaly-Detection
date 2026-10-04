import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class WindowedFlowDataset(Dataset):
    """
    PyTorch Dataset for windowed time-series network flows.
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def create_sliding_windows(features: np.ndarray, labels: np.ndarray, window_size: int):
    """
    Transforms tabular flow features into sequential windows of length `window_size`.
    """
    num_samples = len(features)
    if num_samples < window_size:
        raise ValueError(f"Sample count ({num_samples}) is less than window size ({window_size})")

    sub_windows = [
        features[i : i + window_size]
        for i in range(num_samples - window_size + 1)
    ]
    sequences = np.array(sub_windows, dtype=np.float32)
    target_labels = labels[window_size - 1 :].astype(np.int64)
    return sequences, target_labels


class IoTDataManager:
    """
    Manages loading, scaling, windowing, and partitioning data across virtual IoT instances (fl_n).
    Guarantees temporal continuity within class traffic captures and strict multi-window alignment.
    """
    def __init__(
        self,
        csv_path: str = "dataset/processed_dataset.csv",
        test_size: float = 0.2,
        random_state: int = 42,
        max_samples_per_class: int = 1500,
        max_window_size: int = 40
    ):
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.max_window_size = max_window_size
        self.raw_df = pd.read_csv(csv_path)

        feature_cols = [c for c in self.raw_df.columns if c not in ["label", "label_name"]]
        self.feature_cols = feature_cols
        self.num_features = len(feature_cols)
        self.num_classes = self.raw_df["label"].nunique()

        # Group by label to preserve temporal flow order within captures
        self.train_blocks = []
        self.test_blocks = []
        
        for lbl in sorted(self.raw_df["label"].unique()):
            group = self.raw_df[self.raw_df["label"] == lbl]
            if max_samples_per_class is not None and max_samples_per_class > 0:
                group = group.head(max_samples_per_class)
            n = len(group)
            n_train = int(n * (1.0 - test_size))
            self.train_blocks.append(group.iloc[:n_train])
            self.test_blocks.append(group.iloc[n_train:])

        # Fit scaler strictly on training blocks
        train_raw_X = np.concatenate([b[feature_cols].values for b in self.train_blocks], axis=0)
        self.scaler = StandardScaler()
        self.scaler.fit(train_raw_X)
        self.X_raw = np.concatenate([b[feature_cols].values for b in self.train_blocks + self.test_blocks], axis=0)
        self.y_raw = np.concatenate([b["label"].values for b in self.train_blocks + self.test_blocks], axis=0)

        print(f"IoTDataManager initialized: {len(self.train_blocks)} classes, "
              f"{len(train_raw_X)} train flows, {len(self.X_raw) - len(train_raw_X)} test flows.")

    def get_dataloaders(self, window_size: int, num_clients: int = 3, batch_size: int = 128):
        """
        Returns client loaders (FL), central loader (Non-FL), and aligned test loader.
        """
        # 1. Train sequences generated per temporal block
        all_train_seqs = []
        all_train_lbls = []
        for blk in self.train_blocks:
            feats = self.scaler.transform(blk[self.feature_cols].values)
            lbls = blk["label"].values
            if len(feats) >= window_size:
                seqs, targets = create_sliding_windows(feats, lbls, window_size)
                all_train_seqs.append(seqs)
                all_train_lbls.append(targets)
        train_seqs = np.concatenate(all_train_seqs, axis=0)
        train_lbls = np.concatenate(all_train_lbls, axis=0)

        # 2. Test sequences aligned across ALL window sizes (ending at time t >= max_window_size - 1 per block)
        all_test_seqs = []
        all_test_lbls = []
        for blk in self.test_blocks:
            feats = self.scaler.transform(blk[self.feature_cols].values)
            lbls = blk["label"].values
            start_t = self.max_window_size - 1
            if len(feats) > start_t:
                seqs = np.array([
                    feats[t - window_size + 1 : t + 1]
                    for t in range(start_t, len(feats))
                ], dtype=np.float32)
                targets = lbls[start_t:].astype(np.int64)
                all_test_seqs.append(seqs)
                all_test_lbls.append(targets)
        test_seqs = np.concatenate(all_test_seqs, axis=0)
        test_lbls = np.concatenate(all_test_lbls, axis=0)

        # Centralized non-FL loader
        central_dataset = WindowedFlowDataset(train_seqs, train_lbls)
        central_loader = DataLoader(central_dataset, batch_size=batch_size, shuffle=True)

        # Test loader
        test_dataset = WindowedFlowDataset(test_seqs, test_lbls)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Federated client loaders (partitioning train data across clients)
        n_train = len(train_lbls)
        indices = np.random.RandomState(self.random_state).permutation(n_train)
        chunk_size = n_train // num_clients
        client_loaders = []

        for i in range(num_clients):
            start_i = i * chunk_size
            end_i = (i + 1) * chunk_size if i < num_clients - 1 else n_train
            c_idx = indices[start_i:end_i]
            c_seqs = train_seqs[c_idx]
            c_lbls = train_lbls[c_idx]
            client_loaders.append(
                DataLoader(WindowedFlowDataset(c_seqs, c_lbls), batch_size=batch_size, shuffle=True)
            )

        return client_loaders, central_loader, test_loader, (test_seqs, test_lbls)
