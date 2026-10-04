import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

class RandomForestEnsembler:
    """
    Random Forest Decision Tree Classifier Ensembler (rfc)
    combining prediction probabilities of the 7 global ML models (M_wi)
    as defined in Section III-B(4) and Fig. 3 of the paper.
    """
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def extract_ensemble_features(self, models_dict: dict, eval_data_dict: dict, device: torch.device):
        """
        Extracts probability vectors from each window-size model M_wi.
        models_dict: {window_size: trained_model}
        eval_data_dict: {window_size: (aligned_sequences, aligned_labels)}
        Returns:
            ensemble_features: shape (N_samples, 5 * num_models)
            reference_labels: shape (N_samples,)
        """
        all_probs = []
        reference_labels = None

        for w, model in sorted(models_dict.items()):
            model.eval()
            model.to(device)
            seqs, lbls = eval_data_dict[w]

            if reference_labels is None:
                reference_labels = lbls
            else:
                assert len(lbls) == len(reference_labels), "Length mismatch in aligned sequences"

            with torch.no_grad():
                tensor_seqs = torch.tensor(seqs, dtype=torch.float32).to(device)
                batch_size = 128
                probs_list = []
                for b_start in range(0, len(tensor_seqs), batch_size):
                    batch = tensor_seqs[b_start : b_start + batch_size]
                    logits = model(batch)
                    probs = torch.softmax(logits, dim=-1)
                    probs_list.append(probs.cpu().numpy())
                
                model_probs = np.vstack(probs_list)
                all_probs.append(model_probs)

        # Concatenate probability outputs from all models M_wi: shape (N, 5 * num_models)
        ensemble_features = np.hstack(all_probs)
        return ensemble_features, reference_labels

    def fit(self, X_ensemble: np.ndarray, y: np.ndarray):
        """
        Trains the Random Forest Ensembler on prediction probability vectors.
        """
        self.rfc.fit(X_ensemble, y)
        return self

    def predict(self, X_ensemble: np.ndarray):
        """
        Predicts attack class using trained Random Forest Ensembler.
        """
        return self.rfc.predict(X_ensemble)

    def evaluate(self, X_ensemble: np.ndarray, y_true: np.ndarray, target_names: list = None):
        """
        Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
        """
        y_pred = self.predict(X_ensemble)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "classification_report": report,
            "y_pred": y_pred
        }

    def cross_validate(self, X_ensemble: np.ndarray, y: np.ndarray, n_splits: int = 5):
        """
        Executes K-fold Stratified Cross-Validation as shown in Fig. 10 of the paper.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        acc_scores = []
        prec_scores = []
        rec_scores = []
        f1_scores = []

        for train_idx, val_idx in skf.split(X_ensemble, y):
            clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
            clf.fit(X_ensemble[train_idx], y[train_idx])
            y_pred = clf.predict(X_ensemble[val_idx])

            acc_scores.append(accuracy_score(y[val_idx], y_pred))
            prec_scores.append(precision_score(y[val_idx], y_pred, average="weighted", zero_division=0))
            rec_scores.append(recall_score(y[val_idx], y_pred, average="weighted", zero_division=0))
            f1_scores.append(f1_score(y[val_idx], y_pred, average="weighted", zero_division=0))

        return {
            "mean_accuracy": float(np.mean(acc_scores)),
            "mean_precision": float(np.mean(prec_scores)),
            "mean_recall": float(np.mean(rec_scores)),
            "mean_f1": float(np.mean(f1_scores)),
            "std_accuracy": float(np.std(acc_scores)),
        }
