#!/usr/bin/env python3
"""
Re-collect fusion snapshots with enhanced features and proper LOS/OnlineTune.
This will create richer 11-dimensional feature vectors instead of the current 8D.
"""
import os
import subprocess
import sys

def run_enhanced_collection():
    """Run data collection with enhanced features"""
    
    # Define experiments to collect (training + validation)
    experiments = [
        # Training experiments
        "default_3_random_0",
        "default_3_random2_0", 
        "default_3_random3_0b",
        "default_3_zigzag_0",
        
        # Validation experiments
        "default_3_random3_2",
    ]
    
    # Enhanced collection parameters
    base_cmd = [
        "python", "swarm_target_tracking.py",
        "--use_height", "--use_height_tf",
        "--uwb_std", "0.33",
        "--pair_corr", "0.75", 
        "--sigma_a_xy", "1.0",
        "--sigma_a_z", "0.45",
        "--ci_method", "grid",
        "--ci_objective", "trace",
        "--use_los",  # Enable LOS for dynamic los_score
        "--los_influence", "0.2",
        "--geom_influence", "0.4",
        "--ema_alpha", "0.0",
        "--online_tune",  # Enable OnlineTuner for dynamic nis_ema/gate_sigma
        "--online_r_min_scale", "0.75",
        "--online_r_max_scale", "10.0",
        "--gate_target", "0.97",
        "--gate_sigma_init", "3.0",
        "--q_adapt",
        "--collect_fusion",  # Collect fusion snapshots
        "--out", "outputs_swarm_enhanced"
    ]
    
    print("🚀 Enhanced Feature Collection Started")
    print("=" * 60)
    print("New features will include:")
    print("  8. R_pair       - Tag-pair measurement variance")
    print("  9. m_eff        - Effective number of tag pairs")  
    print(" 10. iqr          - Range measurement IQR")
    print("Plus dynamic LOS scores and OnlineTuner stats!")
    print("=" * 60)
    
    collected_experiments = []
    failed_experiments = []
    
    for exp in experiments:
        target = "ifo003"  # Consistent target for comparison
        
        print(f"\n📊 Collecting: {exp} -> {target}")
        
        cmd = base_cmd + [
            "--exp", exp,
            "--target", target
        ]
        
        try:
            # Run the collection
            result = subprocess.run(
                cmd, 
                cwd="/home/nick/Thesis",
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per experiment
            )
            
            if result.returncode == 0:
                print(f"  ✅ {exp} -> {target}: SUCCESS")
                collected_experiments.append(f"{exp}_{target}")
                
                # Extract key metrics from output
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "Target RMSE" in line:
                        print(f"     {line.strip()}")
                    elif "[COLLECT]" in line and "fusion snaps" in line:
                        print(f"     {line.strip()}")
            else:
                print(f"  ❌ {exp} -> {target}: FAILED")
                print(f"     Error: {result.stderr.strip()}")
                failed_experiments.append(f"{exp}_{target}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {exp} -> {target}: TIMEOUT")
            failed_experiments.append(f"{exp}_{target}")
        except Exception as e:
            print(f"  💥 {exp} -> {target}: EXCEPTION - {e}")
            failed_experiments.append(f"{exp}_{target}")
    
    print("\n" + "=" * 60)
    print("📊 COLLECTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {len(collected_experiments)}")
    for exp in collected_experiments:
        print(f"   - {exp}")
    
    if failed_experiments:
        print(f"\n❌ Failed: {len(failed_experiments)}")
        for exp in failed_experiments:
            print(f"   - {exp}")
    
    # Next steps
    print(f"\n🎯 NEXT STEPS:")
    if len(collected_experiments) >= 4:  # Need at least training experiments
        print("1. Merge enhanced fusion snapshots:")
        print("   python swarm_batch_run.py --merge_fusion_snaps outputs_swarm_enhanced data/fusion_snaps_enhanced.jsonl")
        print("\n2. Train enhanced FusionNet:")
        print("   python -m swarm_ml.train_fusionnet_cli \\")
        print("     --snaps data/fusion_snaps_enhanced.jsonl \\")
        print("     --out models/fusionnet_enhanced \\")
        print("     --split_mode by_exp \\")
        print("     --val_exps default_3_random3_2 \\")
        print("     --epochs 60 --patience 6 --lr 3e-4")
        print("\n3. Compare with baseline:")
        print("   python eval_baselines.py  # Update to use enhanced features")
    else:
        print("❌ Insufficient successful collections. Check errors above.")
    
    return len(collected_experiments), len(failed_experiments)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("🔍 DRY RUN MODE - Commands would be:")
        print("   python swarm_target_tracking.py --exp <exp> --target ifo003 \\")
        print("     --use_los --online_tune --collect_fusion \\")
        print("     --out outputs_swarm_enhanced")
        print("\nRun without --dry-run to execute.")
    else:
        success_count, fail_count = run_enhanced_collection()
        
        if fail_count > 0:
            print(f"\n⚠️  {fail_count} collections failed. Check logs above.")
            sys.exit(1)
        else:
            print(f"\n🎉 All {success_count} collections completed successfully!")
            sys.exit(0)