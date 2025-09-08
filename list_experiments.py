import json
from collections import Counter

def list_experiments(jsonl_path):
    """List all experiments and their sample counts"""
    
    exp_counts = Counter()
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            snap = json.loads(line)
            exp = snap.get('exp', 'unknown')
            exp_counts[exp] += 1
    
    print("Experiments in fusion_snaps_all.jsonl:")
    print("=" * 50)
    
    total_samples = sum(exp_counts.values())
    
    for exp, count in sorted(exp_counts.items()):
        percentage = (count / total_samples) * 100
        status = "🎯 VALIDATION" if exp == "default_3_random3_2" else "📚 TRAINING"
        print(f"{status} {exp}: {count:,} samples ({percentage:.1f}%)")
    
    print(f"\nTotal: {total_samples:,} samples across {len(exp_counts)} experiments")
    
    return exp_counts

# Run analysis
exp_counts = list_experiments('data/fusion_snaps_all.jsonl')