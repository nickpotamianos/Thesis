#!/usr/bin/env python3
"""
Quick enhanced feature analysis for 11D features.
"""
import json
import numpy as np
from collections import defaultdict

def quick_analyze_enhanced(jsonl_file: str, max_samples=10000):
    """Quick analysis with sample limit"""
    
    print(f"📊 Quick Enhanced Feature Analysis")
    print(f"Loading up to {max_samples:,} samples...")
    
    exp_features = defaultdict(list)
    total_loaded = 0
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if total_loaded >= max_samples:
                break
                
            data = json.loads(line.strip())
            exp_id = data['exp_id']
            X = np.array(data['X'])  # Shape: (n_nodes, n_features)
            
            # Add all node features from this snapshot
            for node_feats in X:
                exp_features[exp_id].append(node_feats)
                total_loaded += 1
                
                if total_loaded >= max_samples:
                    break
    
    print(f"✅ Loaded {total_loaded:,} feature vectors from {len(exp_features)} experiments")
    
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
    
    # Convert to arrays and compute stats
    exp_stats = {}
    for exp_id, feats_list in exp_features.items():
        if len(feats_list) > 0:
            X_exp = np.array(feats_list)
            exp_stats[exp_id] = {
                'mean': X_exp.mean(axis=0),
                'std': X_exp.std(axis=0),
                'count': len(feats_list)
            }
    
    # Find validation experiment
    val_exp = None
    train_exps = []
    for exp_id in exp_stats.keys():
        if 'random3_2' in exp_id:
            val_exp = exp_id
        else:
            train_exps.append(exp_id)
    
    print(f"\n🎯 EXPERIMENT BREAKDOWN:")
    for exp_id, stats in exp_stats.items():
        exp_type = "🎯 VALIDATION" if exp_id == val_exp else "📚 TRAINING"
        print(f"{exp_type:15} {exp_id:25} {stats['count']:6,} samples")
    
    if val_exp and train_exps:
        print(f"\n📊 FEATURE COMPARISON (validation vs training):")
        print("-" * 70)
        
        val_stats = exp_stats[val_exp]
        
        # Combine training data
        train_data = []
        for exp in train_exps:
            train_data.extend(exp_features[exp])
        train_array = np.array(train_data)
        train_mean = train_array.mean(axis=0)
        train_std = train_array.std(axis=0)
        
        static_count = 0
        enhanced_feats = []
        
        for i, fname in enumerate(feature_names):
            val_mean = val_stats['mean'][i]
            val_std = val_stats['std'][i]
            
            # Check if static
            is_static = val_std < 1e-6 and train_std[i] < 1e-6
            status = "⚠️ STATIC" if is_static else "✅ Dynamic"
            
            if is_static:
                static_count += 1
            
            # Mark enhanced features
            if i >= 8:  # R_pair, m_eff, iqr
                enhanced_feats.append((fname, val_mean, val_std, train_mean[i], train_std[i], status))
            
            print(f"{fname:12} | Val: {val_mean:8.3f} ± {val_std:6.3f} | Train: {train_mean[i]:8.3f} ± {train_std[i]:6.3f} | {status}")
        
        print(f"\n🔍 ENHANCED FEATURES ANALYSIS:")
        print("-" * 50)
        for fname, val_m, val_s, train_m, train_s, status in enhanced_feats:
            print(f"{fname}:")
            print(f"  📊 Validation: {val_m:.4f} ± {val_s:.4f}")
            print(f"  📊 Training:   {train_m:.4f} ± {train_s:.4f}")
            
            # Check discriminative power
            if val_s > 1e-4 and train_s > 1e-4:
                unique_approx = "High variance - good discrimination"
            else:
                unique_approx = "Low variance - limited discrimination"
            print(f"  🎯 Assessment: {unique_approx}")
            print()
        
        print(f"📋 SUMMARY:")
        print(f"✅ Total features: {len(feature_names)}")
        print(f"⚠️  Static features: {static_count}")
        print(f"🚀 Enhanced features: 3 (R_pair, m_eff, iqr)")
        print(f"🎯 Sample analyzed: {total_loaded:,} / {max_samples:,}")
        
        if static_count <= 3:
            print(f"🎉 Good feature quality for enhanced FusionNet!")
        else:
            print(f"⚠️  Many static features detected")
    else:
        print("❌ Could not find validation experiment for comparison")

if __name__ == "__main__":
    import sys
    jsonl_file = sys.argv[1] if len(sys.argv) > 1 else 'data/fusion_snaps_enhanced.jsonl'
    quick_analyze_enhanced(jsonl_file)