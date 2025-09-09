#!/usr/bin/env python3
# debug_biasnet_simple.py
# Simple BiasNet feature and prediction test to isolate issues

import json
import torch
import numpy as np
from swarm_ml.models import BiasNet

def main():
    # Load a few training samples
    samples_file = "test_biasnet_fix_output/collect/merged_train_samples.jsonl"
    samples = []
    with open(samples_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Just first 10 samples
                break
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} sample(s)")
    
    # Load trained BiasNet
    model_dir = "test_biasnet_fix_output/models/biasnet_fixed"
    with open(f"{model_dir}/biasnet_meta.json", "r") as f:
        meta = json.load(f)
    
    in_dim = int(meta["in_dim"])
    print(f"BiasNet input dimension: {in_dim}")
    print("Expected feature structure: [uwb_range, 0,0,0,0, los_score, dz, 0,0, R_pair, m_eff, iqr, onehot...]")
    print(f"Expected total dimension with 2 trackers: 9 + 3 + 2 = 14")
    
    model = BiasNet(in_dim)
    state = torch.load(f"{model_dir}/biasnet.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    
    print("\nSample analysis:")
    for i, sample in enumerate(samples):
        features = np.array(sample["features"], dtype=np.float32)
        true_bias = sample["bias"]
        
        # Predict bias
        with torch.no_grad():
            pred_bias = model(torch.from_numpy(features.reshape(1, -1))).item()
        
        print(f"Sample {i+1}:")
        print(f"  Features: {features}")
        print(f"  True bias: {true_bias:.4f}m")
        print(f"  Pred bias: {pred_bias:.4f}m")
        print(f"  Error: {abs(pred_bias - true_bias):.4f}m")
        print()
    
    # Test on a synthetic sample similar to test conditions  
    print("Testing synthetic sample (similar to test data):")
    # Create a feature vector similar to test experiment with enhanced features
    synthetic_features = np.array([
        2.0,     # uwb_range
        0.0, 0.0, 0.0, 0.0,  # geometry (no prediction available)
        0.5,     # los_score (neutral)
        -0.03,   # height_diff (similar to training)
        0.0, 0.0, # residual_hist (no history)
        0.2,     # R_pair (typical pair variance)
        2.0,     # m_eff (typical pair count)
        0.05,    # iqr (typical range dispersion)
        1.0, 0.0 # onehot for tracker (assuming 2 trackers)
    ], dtype=np.float32)
    
    with torch.no_grad():
        synthetic_pred = model(torch.from_numpy(synthetic_features.reshape(1, -1))).item()
    
    print(f"Synthetic features: {synthetic_features}")
    print(f"Synthetic prediction: {synthetic_pred:.4f}m")
    
    # Check if model is making reasonable predictions
    all_true_biases = [s["bias"] for s in samples]
    all_pred_biases = []
    for s in samples:
        features = np.array(s["features"], dtype=np.float32)
        with torch.no_grad():
            pred = model(torch.from_numpy(features.reshape(1, -1))).item()
        all_pred_biases.append(pred)
    
    print(f"\nSummary statistics:")
    print(f"True bias - mean: {np.mean(all_true_biases):.4f}, std: {np.std(all_true_biases):.4f}")
    print(f"Pred bias - mean: {np.mean(all_pred_biases):.4f}, std: {np.std(all_pred_biases):.4f}")
    print(f"MAE: {np.mean([abs(t-p) for t,p in zip(all_true_biases, all_pred_biases)]):.4f}")

if __name__ == "__main__":
    main()