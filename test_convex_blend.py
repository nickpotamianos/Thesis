#!/usr/bin/env python3
"""
Simple debug script to verify BiasNet bias blending behavior
"""
import numpy as np

def test_convex_blend():
    """Test the convex blending formula"""
    # Simulate typical values
    bias_online = -0.08  # EMA learned bias
    bias_model = -0.06   # BiasNet prediction
    gamma = 0.6          # bias_gain
    
    # Convex blend (new, correct)
    bias_convex = gamma * bias_model + (1.0 - gamma) * bias_online
    
    # Additive blend (old, incorrect)  
    bias_additive = bias_model * gamma + bias_online
    
    print(f"=== Bias Blending Test ===")
    print(f"bias_online: {bias_online:.4f}")
    print(f"bias_model:  {bias_model:.4f}")
    print(f"gamma:       {gamma:.2f}")
    print(f"")
    print(f"Convex (correct):   {bias_convex:.4f}")
    print(f"Additive (wrong):   {bias_additive:.4f}")
    print(f"")
    print(f"Difference: {abs(bias_convex - bias_additive):.4f}")
    
    # Expected: convex should be closer to 0, additive should overcorrect
    print(f"")
    print(f"With gamma=0.6:")
    print(f"  Convex:   0.6 * {bias_model:.3f} + 0.4 * {bias_online:.3f} = {bias_convex:.4f}")
    print(f"  Additive: {gamma:.1f} * {bias_model:.3f} + {bias_online:.3f} = {bias_additive:.4f}")

if __name__ == "__main__":
    test_convex_blend()