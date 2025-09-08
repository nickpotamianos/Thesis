#!/usr/bin/env python3
"""
Merge enhanced fusion snapshots from multiple experiments.
"""
import json
import os
import glob

def merge_fusion_snapshots(input_dir: str, output_file: str):
    """Merge all fusion_snaps.jsonl files from subdirectories"""
    
    # Find all fusion_snaps.jsonl files
    pattern = os.path.join(input_dir, "*/fusion_snaps.jsonl")
    snapshot_files = glob.glob(pattern)
    
    if not snapshot_files:
        print(f"❌ No fusion_snaps.jsonl files found in {input_dir}")
        return 0
    
    print(f"🔄 Merging {len(snapshot_files)} snapshot files...")
    
    total_count = 0
    with open(output_file, 'w') as outf:
        for snap_file in sorted(snapshot_files):
            exp_name = os.path.basename(os.path.dirname(snap_file))
            print(f"   📂 {exp_name}: ", end="", flush=True)
            
            count = 0
            with open(snap_file, 'r') as inf:
                for line in inf:
                    line = line.strip()
                    if line:
                        # Parse, add experiment ID, and write
                        data = json.loads(line)
                        data['exp_id'] = exp_name
                        outf.write(json.dumps(data) + '\n')
                        count += 1
            
            print(f"{count:,} snapshots")
            total_count += count
    
    print(f"✅ Merged {total_count:,} total snapshots -> {output_file}")
    return total_count

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python merge_enhanced_snapshots.py <input_dir> <output_file>")
        print("Example: python merge_enhanced_snapshots.py outputs_swarm_enhanced data/fusion_snaps_enhanced.jsonl")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    count = merge_fusion_snapshots(input_dir, output_file)
    
    if count > 0:
        print(f"\n🎯 Ready for enhanced FusionNet training!")
        print(f"   Training command:")
        print(f"   python -m swarm_ml.train_fusionnet_cli \\")
        print(f"     --snaps {output_file} \\")
        print(f"     --out models/fusionnet_enhanced \\")
        print(f"     --split_mode by_exp \\")
        print(f"     --val_exps default_3_random3_2_ifo003 \\")
        print(f"     --epochs 60 --patience 6 --lr 3e-4")
    else:
        print("❌ No snapshots to merge")
        sys.exit(1)