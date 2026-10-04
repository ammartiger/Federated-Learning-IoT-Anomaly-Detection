import os
import sys
import time
import json
import argparse
import pandas as pd
import numpy as np
import torch

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from models import create_model, GRU_CONFIGS
from dataset_loader import IoTDataManager
from federated import train_federated, train_centralized
from ensembler import RandomForestEnsembler
from plot_results import (
    plot_gru_comparison,
    plot_training_times,
    plot_ensembler_results,
    plot_overall_average
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_CLASSES = ["Clean", "MITM", "ModbusQueryFlood", "PingDDoS", "SYNDDoS"]

def run_replication(
    models_to_run: list = None,
    windows_to_run: list = None,
    rounds: int = 5,
    local_epochs: int = 1,
    lr: float = 0.003,
    num_clients: int = 3,
    batch_size: int = 128,
    max_samples_per_class: int = 1500,
    device: str = "cpu"
):
    if models_to_run is None:
        models_to_run = ["GRU-1", "GRU-2", "GRU-3", "GRU-4"]
    if windows_to_run is None:
        windows_to_run = [1, 5, 10, 15, 20, 30, 40]

    print("=" * 75)
    print(" REPLICATING IEEE IoT JOURNAL PAPER:")
    print(" 'Federated-Learning-Based Anomaly Detection for IoT Security Attacks'")
    print("=" * 75)
    print(f"Models to evaluate : {models_to_run}")
    print(f"Window sizes       : {windows_to_run}")
    print(f"FL Clients         : {num_clients} virtual IoT devices (fl_1 ... fl_{num_clients})")
    print(f"FL Rounds          : {rounds} (with {local_epochs} local epochs/round)")
    print(f"Batch size         : {batch_size}")
    print(f"Device             : {device}")
    print("-" * 75)

    # 1. Load and prepare dataset
    print("\n[Step 1] Loading Modbus dataset and preparing flow sequences...")
    data_manager = IoTDataManager(
        csv_path="dataset/processed_dataset.csv",
        test_size=0.2,
        random_state=42,
        max_samples_per_class=max_samples_per_class
    )
    input_size = data_manager.num_features
    num_classes = data_manager.num_classes
    print(f"Dataset ready: {data_manager.X_raw.shape[0]} total flows, {input_size} features, {num_classes} classes.")

    all_results = {
        "models": {},
        "training_times": {},
        "ensembler": {},
        "overall_summary": {}
    }

    fl_accuracies_all = []
    non_fl_accuracies_all = []

    # Store trained FL models for the Ensembler (keyed by model_name and window_size)
    fl_global_models = {m: {} for m in models_to_run}
    eval_data_cache = {}

    # 2. Train and evaluate each GRU model across all window sizes
    for model_name in models_to_run:
        print(f"\n=======================================================")
        print(f"   EVALUATING {model_name} (Layers: {GRU_CONFIGS[model_name]['num_layers']}, "
              f"Hidden: {GRU_CONFIGS[model_name]['hidden_size']}, Dropout: {GRU_CONFIGS[model_name]['dropout']})")
        print(f"=======================================================")

        all_results["models"][model_name] = {"FL": {}, "Non-FL": {}}
        all_results["training_times"][model_name] = {"FL": {}, "Non-FL": {}}

        for w in windows_to_run:
            print(f"\n---> Model: {model_name} | Window Size W = {w}")

            # Get loaders
            client_loaders, central_train_loader, test_loader, (test_seqs, test_lbls) = data_manager.get_dataloaders(
                window_size=w,
                num_clients=num_clients,
                batch_size=batch_size
            )
            eval_data_cache[w] = (test_seqs, test_lbls)

            # Template model
            template = create_model(model_name, input_size=input_size, num_classes=num_classes)

            # --- A. Federated Learning (FL) ---
            print(f"  [FL] Training across {num_clients} virtual clients + flaverage...")
            t0 = time.time()
            fl_model, fl_hist, fl_time = train_federated(
                model_template=template,
                client_loaders=client_loaders,
                test_loader=test_loader,
                rounds=rounds,
                local_epochs=local_epochs,
                lr=lr,
                device=device
            )
            fl_metrics = fl_hist[-1]
            fl_global_models[model_name][w] = fl_model

            print(f"  [FL] Completed in {fl_time:.2f}s | Acc: {fl_metrics['accuracy']*100:.2f}% | "
                  f"F1: {fl_metrics['f1']*100:.2f}% | Prec: {fl_metrics['precision']*100:.2f}% | "
                  f"Rec: {fl_metrics['recall']*100:.2f}%")

            # --- B. Centralized Non-FL Baseline ---
            print(f"  [Non-FL] Training centralized baseline model...")
            non_fl_model, non_fl_hist, non_fl_time = train_centralized(
                model_template=template,
                central_train_loader=central_train_loader,
                test_loader=test_loader,
                epochs=rounds * local_epochs,
                lr=lr,
                device=device
            )
            non_fl_metrics = non_fl_hist[-1]

            print(f"  [Non-FL] Completed in {non_fl_time:.2f}s | Acc: {non_fl_metrics['accuracy']*100:.2f}% | "
                  f"F1: {non_fl_metrics['f1']*100:.2f}% | Prec: {non_fl_metrics['precision']*100:.2f}% | "
                  f"Rec: {non_fl_metrics['recall']*100:.2f}%")

            # Store metrics
            all_results["models"][model_name]["FL"][str(w)] = {
                "accuracy": fl_metrics["accuracy"],
                "precision": fl_metrics["precision"],
                "recall": fl_metrics["recall"],
                "f1": fl_metrics["f1"],
            }
            all_results["models"][model_name]["Non-FL"][str(w)] = {
                "accuracy": non_fl_metrics["accuracy"],
                "precision": non_fl_metrics["precision"],
                "recall": non_fl_metrics["recall"],
                "f1": non_fl_metrics["f1"],
            }
            all_results["training_times"][model_name]["FL"][str(w)] = fl_time
            all_results["training_times"][model_name]["Non-FL"][str(w)] = non_fl_time

            fl_accuracies_all.append(fl_metrics["accuracy"])
            non_fl_accuracies_all.append(non_fl_metrics["accuracy"])

        # Plot figure for this GRU model (Replicating Figs. 5, 6, 7, 8)
        fig_num = {"GRU-1": 5, "GRU-2": 6, "GRU-3": 7, "GRU-4": 8}[model_name]
        fig_path = os.path.join(RESULTS_DIR, f"fig{fig_num}_{model_name.lower().replace('-', '')}_evaluation.png")
        plot_gru_comparison(
            model_name=model_name,
            fl_metrics=all_results["models"][model_name]["FL"],
            non_fl_metrics=all_results["models"][model_name]["Non-FL"],
            save_path=fig_path
        )

    # 3. Plot Training Times (Replicating Fig. 9)
    print("\n[Step 3] Generating Training Time comparison plot (Fig. 9)...")
    fig9_path = os.path.join(RESULTS_DIR, "fig9_training_times.png")
    plot_training_times(all_results["training_times"], fig9_path)

    # 4. Train and Evaluate Ensembler (Random Forest Classifier on 7 Global Models)
    # The paper ensembled the 7 window sizes of the best GRU model (or across models)
    primary_model = "GRU-1" if "GRU-1" in fl_global_models else models_to_run[0]
    print(f"\n[Step 4] Building Random Forest Ensembler (rfc) over 7 Global Models ({primary_model})...")

    ensembler = RandomForestEnsembler(n_estimators=100, random_state=42)
    dev = torch.device(device)

    # Extract ensemble probability features
    X_ensemble, y_ensemble = ensembler.extract_ensemble_features(
        models_dict=fl_global_models[primary_model],
        eval_data_dict=eval_data_cache,
        device=dev
    )
    print(f"Extracted ensemble probability feature space: {X_ensemble.shape}")

    # Split for ensembler evaluation
    from sklearn.model_selection import train_test_split
    X_ens_train, X_ens_test, y_ens_train, y_ens_test = train_test_split(
        X_ensemble, y_ensemble, test_size=0.3, random_state=42, stratify=y_ensemble
    )

    ensembler.fit(X_ens_train, y_ens_train)
    ens_test_results = ensembler.evaluate(X_ens_test, y_ens_test, target_names=TARGET_CLASSES)

    print("\n--- ENSEMBLER EVALUATION RESULTS ---")
    print(f"Ensemble Accuracy  : {ens_test_results['accuracy']*100:.2f}%")
    print(f"Ensemble F1-Score  : {ens_test_results['f1']*100:.2f}%")
    print(f"Ensemble Precision : {ens_test_results['precision']*100:.2f}%")
    print(f"Ensemble Recall    : {ens_test_results['recall']*100:.2f}%")
    print("\nClassification Report:")
    print(ens_test_results["classification_report"])

    # Cross-validation across window sizes (Fig. 10)
    print("\n[Step 5] Running Cross-Validation across window configurations (Fig. 10)...")
    cv_dict = {}
    for w in windows_to_run:
        # Evaluate ensembler subset using models up to window w
        sub_models = {k: v for k, v in fl_global_models[primary_model].items() if k <= w}
        X_sub, y_sub = ensembler.extract_ensemble_features(sub_models, eval_data_cache, dev)
        cv_res = ensembler.cross_validate(X_sub, y_sub, n_splits=5)
        cv_dict[str(w)] = {
            "accuracy": cv_res["mean_accuracy"],
            "f1": cv_res["mean_f1"],
            "precision": cv_res["mean_precision"],
            "recall": cv_res["mean_recall"],
        }
        print(f"  Window Size {w:2d} -> CV Acc: {cv_res['mean_accuracy']*100:.2f}%, F1: {cv_res['mean_f1']*100:.2f}%")

    all_results["ensembler"] = cv_dict
    fig10_path = os.path.join(RESULTS_DIR, "fig10_cross_validation.png")
    plot_ensembler_results(cv_dict, fig10_path)

    # 5. Overall Average Accuracy (Fig. 11)
    fl_avg_acc = float(np.mean(fl_accuracies_all))
    non_fl_avg_acc = float(np.mean(non_fl_accuracies_all))
    ens_avg_acc = float(np.mean([v["accuracy"] for v in cv_dict.values()]))

    all_results["overall_summary"] = {
        "fl_average_accuracy": fl_avg_acc,
        "non_fl_average_accuracy": non_fl_avg_acc,
        "ensembler_average_accuracy": ens_avg_acc,
    }

    print(f"\n=======================================================")
    print(f"   OVERALL REPLICATION SUMMARY (Fig. 11)")
    print(f"=======================================================")
    print(f"  Federated Learning (FL) Average Accuracy : {fl_avg_acc*100:.2f}% (Paper: ~90.26%)")
    print(f"  Non-FL Centralized Average Accuracy      : {non_fl_avg_acc*100:.2f}% (Paper: ~86.13%)")
    print(f"  FL + Ensembler (RFC) Average Accuracy    : {ens_avg_acc*100:.2f}% (Paper: ~99.50%)")
    print(f"=======================================================")

    fig11_path = os.path.join(RESULTS_DIR, "fig11_average_accuracy.png")
    plot_overall_average(fl_avg_acc, non_fl_avg_acc, ens_avg_acc, fig11_path)

    # Save summary JSON and CSV
    with open(os.path.join(RESULTS_DIR, "replication_summary.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    # Create CSV table of all model results
    table_rows = []
    for m in models_to_run:
        for w in windows_to_run:
            fl_m = all_results["models"][m]["FL"][str(w)]
            nfl_m = all_results["models"][m]["Non-FL"][str(w)]
            table_rows.append({
                "Model": m,
                "Window_Size": w,
                "FL_Accuracy": fl_m["accuracy"],
                "FL_F1": fl_m["f1"],
                "FL_Precision": fl_m["precision"],
                "FL_Recall": fl_m["recall"],
                "FL_Time_Sec": all_results["training_times"][m]["FL"][str(w)],
                "NonFL_Accuracy": nfl_m["accuracy"],
                "NonFL_F1": nfl_m["f1"],
                "NonFL_Precision": nfl_m["precision"],
                "NonFL_Recall": nfl_m["recall"],
                "NonFL_Time_Sec": all_results["training_times"][m]["Non-FL"][str(w)],
            })
    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(os.path.join(RESULTS_DIR, "metrics_table.csv"), index=False)
    print(f"\nSaved metrics table to {os.path.join(RESULTS_DIR, 'metrics_table.csv')}")
    print(f"Saved complete summary JSON to {os.path.join(RESULTS_DIR, 'replication_summary.json')}")
    print("\nREPLICATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate IoT Federated Learning Anomaly Detection paper results")
    parser.add_argument("--models", nargs="+", default=["GRU-1", "GRU-2", "GRU-3", "GRU-4"], help="Models to evaluate")
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 5, 10, 15, 20, 30, 40], help="Window sizes")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL communication rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of virtual IoT clients")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_samples_per_class", type=int, default=1500, help="Class sample cap for fast balanced training")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")

    args = parser.parse_args()
    run_replication(
        models_to_run=args.models,
        windows_to_run=args.windows,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples_per_class,
        device=args.device
    )
