import json
import numpy as np
from collections import defaultdict

def analyze_experiment_features(jsonl_path, target_exp='default_3_random3_2'):
    """Compare enhanced 11D feature statistics between experiments"""
    
    # Load all snapshots
    print("Loading enhanced snapshots...")
    snapshots_by_exp = defaultdict(list)
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            
            # Use 'exp' field for experiment name
            exp_id = data.get('exp', 'unknown')
            
            # X is a list of lists: [[node1_features], [node2_features], ...]
            # Each inner list has 11 features
            X_list = data['X']
            
            # Convert to numpy array directly - X is already in the right shape
            X = np.array(X_list)
            
            # Verify it's the expected shape (n_nodes, 11)
            if X.ndim != 2 or X.shape[1] != 11:
                print(f"Warning: Unexpected X shape {X.shape}, expected (n_nodes, 11)")
                continue
            
            # Store each node's features separately
            for node_feats in X:
                snapshots_by_exp[exp_id].append(node_feats)
    
    # Convert to arrays
    exp_features = {}
    for exp_id, feats_list in snapshots_by_exp.items():
        if feats_list:  # Only if we have features
            exp_features[exp_id] = np.array(feats_list)
    
    # Compute statistics per experiment
    print(f"Analyzing {len(exp_features)} experiments...")
    print(f"Experiments found: {list(exp_features.keys())[:10]}...")  # Show first 10
    
    stats = {}
    for exp, features_array in exp_features.items():
        if len(features_array) == 0:
            continue
        
        # features_array is already shape (n_samples, n_features)
        stats[exp] = {
            'mean': features_array.mean(axis=0),
            'std': features_array.std(axis=0),
            'min': features_array.min(axis=0),
            'max': features_array.max(axis=0),
            'n_samples': len(features_array)
        }
    
    # Feature names for enhanced 11D features
    feature_names = [
        'var_pos',      # 0
        'reliability',  # 1  
        'z_agg',        # 2
        'R_eff',        # 3
        'geom_ez',      # 4
        'los_score',    # 5
        'gate_sigma',   # 6
        'nis_ema',      # 7
        'R_pair',       # 8  <- NEW
        'm_eff',        # 9  <- NEW  
        'iqr'           # 10 <- NEW
    ]
    
    # Global statistics across ALL experiments
    print("\n" + "="*70)
    print("GLOBAL STATISTICS (all experiments combined):")
    print("="*70)
    
    all_features = []
    for features_array in exp_features.values():
        all_features.append(features_array)
    
    if all_features:
        all_features_stacked = np.vstack(all_features)
        
        print(f"Total samples: {len(all_features_stacked):,}")
        print(f"\n{'Feature':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-"*60)
        
        for i, fname in enumerate(feature_names):
            feat_vals = all_features_stacked[:, i]
            print(f"{fname:<15} {np.mean(feat_vals):>10.4f} {np.std(feat_vals):>10.4f} "
                  f"{np.min(feat_vals):>10.4f} {np.max(feat_vals):>10.4f}")
    
    # Check for problematic features
    print("\n" + "="*70)
    print("PROBLEMATIC FEATURES CHECK:")
    print("="*70)
    
    for i, fname in enumerate(feature_names):
        feat_vals = all_features_stacked[:, i]
        std_val = np.std(feat_vals)
        
        if std_val < 0.01:
            print(f"⚠️  {fname}: Near-constant (std={std_val:.6f})")
        elif std_val < 0.1:
            print(f"⚡ {fname}: Low variance (std={std_val:.4f})")
    
    # Compare specific experiment to others if requested
    if target_exp in stats:
        print(f"\n" + "="*70)
        print(f"TARGET EXPERIMENT ANALYSIS: {target_exp}")
        print("="*70)
        print(f"Samples in target: {stats[target_exp]['n_samples']}\n")
        
        val_stats = stats[target_exp]
        
        # Compute stats for all OTHER experiments combined
        other_features = []
        for exp, features_array in exp_features.items():
            if exp != target_exp:
                other_features.append(features_array)
        
        if other_features:
            X_others = np.vstack(other_features)
            others_mean = X_others.mean(axis=0)
            others_std = X_others.std(axis=0)
            
            print("Feature comparison (target vs others):")
            print("-" * 60)
            
            for i, fname in enumerate(feature_names):
                val_mean = val_stats['mean'][i]
                val_std = val_stats['std'][i]
                other_mean = others_mean[i]
                other_std = others_std[i]
                
                # How many standard deviations apart are the means?
                if other_std > 1e-6:
                    z_score = abs(val_mean - other_mean) / other_std
                    if z_score > 3:
                        status = "SEVERE DIFF"
                    elif z_score > 1.5:
                        status = "Different"
                    else:
                        status = "Similar"
                else:
                    status = "No variance"
                
                print(f"{fname:12} | Target: {val_mean:7.3f}±{val_std:5.3f} | "
                      f"Others: {other_mean:7.3f}±{other_std:5.3f} | {status}")
    else:
        print(f"\nTarget experiment '{target_exp}' not found in data!")
        print(f"Available experiments: {list(stats.keys())[:5]}...")
    
    return stats

# Run analysis
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_fusion.py <fusion_snaps.jsonl> [target_exp]")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    target_exp = sys.argv[2] if len(sys.argv) > 2 else 'default_3_random3_2'
    
    stats = analyze_experiment_features(jsonl_file, target_exp)