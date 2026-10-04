import os
import zipfile
import time
import pandas as pd
import numpy as np
from scapy.all import PcapReader
from cicflowmeter.flow_session import FlowSession

ZIP_PATH = "dataset/captures1_v2.zip"
OUTPUT_CSV = "dataset/extracted_flows.csv"
PROCESSED_CSV = "dataset/processed_dataset.csv"

# Select representative pcaps for each of the 5 classes
PCAP_CONFIG = [
    # Clean / Benign
    ("Clean", "captures1_v2/clean/eth2dump-clean-0,5h_1.pcap", 30000),
    ("Clean", "captures1_v2/clean/eth2dump-clean-1h_1.pcap", 30000),
    
    # MITM
    ("MITM", "captures1_v2/mitm/eth2dump-mitm-change-1m-0,5h_1.pcap", 20000),
    ("MITM", "captures1_v2/mitm/eth2dump-mitm-change-5m-0,5h_1.pcap", 20000),
    ("MITM", "captures1_v2/mitm/eth2dump-mitm-change-15m-0,5h_1.pcap", 20000),
    ("MITM", "captures1_v2/mitm/eth2dump-mitm-change-1m-1h_1.pcap", 20000),
    
    # Modbus Query Flood
    ("ModbusQueryFlood", "captures1_v2/modbusQueryFlooding/eth2dump-modbusQueryFlooding1m-0,5h_1.pcap", 20000),
    ("ModbusQueryFlood", "captures1_v2/modbusQueryFlooding/eth2dump-modbusQueryFlooding5m-0,5h_1.pcap", 20000),
    ("ModbusQueryFlood", "captures1_v2/modbusQuery2Flooding/eth2dump-modbusQuery2Flooding1m-0,5h_1.pcap", 20000),
    
    # Ping DDoS (ICMP)
    ("PingDDoS", "captures1_v2/pingFloodDDoS/eth2dump-pingFloodDDoS1m-0,5h_1.pcap", 25000),
    ("PingDDoS", "captures1_v2/pingFloodDDoS/eth2dump-pingFloodDDoS5m-0,5h_1.pcap", 25000),
    ("PingDDoS", "captures1_v2/pingFloodDDoS/eth2dump-pingFloodDDoS1m-1h_1.pcap", 25000),
    ("PingDDoS", "captures1_v2/pingFloodDDoS/eth2dump-pingFloodDDoS15m-0,5h_1.pcap", 25000),
    
    # SYN DDoS
    ("SYNDDoS", "captures1_v2/tcpSYNFloodDDoS/eth2dump-tcpSYNFloodDDoS1m-0,5h_1.pcap", 15000),
    ("SYNDDoS", "captures1_v2/tcpSYNFloodDDoS/eth2dump-tcpSYNFloodDDoS1m-1h_1.pcap", 15000),
]

def extract_all():
    os.makedirs("dataset/temp_flows", exist_ok=True)
    all_dfs = []
    
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for idx, (label, pcap_path, max_pkts) in enumerate(PCAP_CONFIG):
            t0 = time.time()
            temp_out = f"dataset/temp_flows/flow_{idx}_{label}.csv"
            FlowSession.output_mode = "csv"
            FlowSession.output = temp_out
            FlowSession.fields = None
            FlowSession.verbose = False
            
            session = FlowSession()
            try:
                with z.open(pcap_path) as f:
                    with PcapReader(f) as pr:
                        for pkt_cnt, pkt in enumerate(pr):
                            if pkt_cnt >= max_pkts:
                                break
                            session.on_packet_received(pkt)
                session.garbage_collect(None)
                del session.output_writer
                
                if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
                    df = pd.read_csv(temp_out)
                    if not df.empty:
                        df["label"] = label
                        all_dfs.append(df)
                        print(f"[{idx+1}/{len(PCAP_CONFIG)}] {label:18s} ({os.path.basename(pcap_path)}): {df.shape[0]} flows ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"Error reading {pcap_path}: {e}")

    if not all_dfs:
        raise RuntimeError("No flows extracted!")
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTotal raw flows extracted: {combined_df.shape}")
    print("Class distribution:")
    print(combined_df["label"].value_counts())
    
    # Preprocess according to the paper
    preprocess(combined_df)

def preprocess(df):
    print("\n--- Preprocessing Features according to Paper ---")
    # Features that do not add value to the learning process (identifiers, timestamps, non-numerics)
    cols_to_drop = [
        "src_ip", "dst_ip", "src_port", "dst_port", "timestamp",
        "flow_id", "Unnamed: 0"
    ]
    features_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Separate label
    labels = features_df["label"]
    numeric_df = features_df.drop(columns=["label"])
    
    # Replace infinities and NaNs
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median())
    numeric_df = numeric_df.fillna(0.0)
    
    # Drop constant columns (zero variance)
    var = numeric_df.var(numeric_only=True)
    non_constant_cols = var[var > 1e-6].index.tolist()
    numeric_df = numeric_df[non_constant_cols]
    print(f"Remaining numerical features after variance cleaning: {numeric_df.shape[1]}")
    
    # Map labels to integers
    label_map = {
        "Clean": 0,
        "MITM": 1,
        "ModbusQueryFlood": 2,
        "PingDDoS": 3,
        "SYNDDoS": 4
    }
    encoded_labels = labels.map(label_map)
    
    processed_df = numeric_df.copy()
    processed_df["label"] = encoded_labels
    processed_df["label_name"] = labels
    
    processed_df.to_csv(PROCESSED_CSV, index=False)
    print(f"Saved processed dataset to {PROCESSED_CSV} ({processed_df.shape})")
    print(processed_df["label_name"].value_counts())

if __name__ == "__main__":
    extract_all()
