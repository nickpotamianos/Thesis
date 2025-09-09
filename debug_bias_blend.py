#!/usr/bin/env python3
"""
Debug script to examine bias blending during tracking
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_bias_meta(output_dir):
    """Analyze bias metadata from tracking output"""
    meta_file = Path(output_dir) / "meta.jsonl"
    if not meta_file.exists():
        print(f"No meta file found at {meta_file}")
        return
    
    bias_online = []
    bias_model = []
    bias_final = []
    bias_gamma = []
    
    with open(meta_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'uwb' in data:
                for uwb_data in data['uwb']:
                    if 'bias_online' in uwb_data and 'bias_model' in uwb_data:
                        bias_online.append(uwb_data['bias_online'])
                        bias_model.append(uwb_data['bias_model'])
                        bias_final.append(uwb_data['bias'])
                        bias_gamma.append(uwb_data.get('bias_gamma', 0.0))
    
    if not bias_online:
        print("No bias data found in meta.jsonl")
        return
        
    bias_online = np.array(bias_online)
    bias_model = np.array(bias_model)
    bias_final = np.array(bias_final)
    bias_gamma = np.array(bias_gamma)
    
    # Expected convex blend: bias_final = gamma * bias_model + (1-gamma) * bias_online
    expected_blend = bias_gamma * bias_model + (1 - bias_gamma) * bias_online
    blend_error = np.abs(bias_final - expected_blend)
    
    print(f"=== Bias Blending Analysis ===")
    print(f"Total samples: {len(bias_online)}")
    print(f"Bias online: mean={bias_online.mean():.4f}, std={bias_online.std():.4f}")
    print(f"Bias model:  mean={bias_model.mean():.4f}, std={bias_model.std():.4f}")
    print(f"Bias final:  mean={bias_final.mean():.4f}, std={bias_final.std():.4f}")
    print(f"Gamma:       mean={bias_gamma.mean():.4f}, std={bias_gamma.std():.4f}")
    print(f"Expected blend: mean={expected_blend.mean():.4f}, std={expected_blend.std():.4f}")
    print(f"Blend error: mean={blend_error.mean():.6f}, max={blend_error.max():.6f}")
    
    # Check for double correction pattern
    # In old code: bias_final ≈ bias_model + bias_online (additive)
    # In new code: bias_final ≈ gamma * bias_model + (1-gamma) * bias_online (convex)
    additive_pred = bias_model + bias_online
    additive_error = np.abs(bias_final - additive_pred)
    
    print(f"\n=== Double Correction Check ===")
    print(f"If additive (old): mean_error={additive_error.mean():.6f}")
    print(f"If convex (new):   mean_error={blend_error.mean():.6f}")
    
    if blend_error.mean() < additive_error.mean():
        print("✓ Convex blending is working correctly")
    else:
        print("✗ Still using additive blending (double correction)")
    
    # Sample comparison
    print(f"\n=== Sample Values ===")
    for i in range(min(5, len(bias_online))):
        print(f"Sample {i+1}:")
        print(f"  online={bias_online[i]:.4f}, model={bias_model[i]:.4f}, gamma={bias_gamma[i]:.2f}")
        print(f"  final={bias_final[i]:.4f}, expected={expected_blend[i]:.4f}, error={blend_error[i]:.6f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debug_bias_blend.py <output_dir>")
        print("Example: python debug_bias_blend.py test_biasnet_fix_output/eval/biasnet/default_3_random3_2_ifo001")
        sys.exit(1)
    
    analyze_bias_meta(sys.argv[1])