#!/usr/bin/env python3
"""
Aggregate all cross-validation results from all folds and scenarios into a single CSV.
"""

import pandas as pd
import glob
from pathlib import Path

def aggregate_cv_results():
    """Collect all CV results into a single comprehensive CSV."""
    
    # Base directory for CV results
    cv_base = Path("runs/20250923_004753_notebook_cv_eval/cv_nonzig")
    
    # Scenario mapping for cleaner names
    scenario_names = {
        "A_baseline_grid": "Baseline_Grid",
        "B1_bn_by_time": "BiasNet_ByTime", 
        "B2_bn_by_exp": "BiasNet_ByExp",
        "C1_fn_by_exp": "FusionNet_ByExp",
        "C2_fn_by_time": "FusionNet_ByTime", 
        "D_bnfn_by_exp": "BiasNet+FusionNet_ByExp",
        "E_budget_k2": "Budgeted_K2"
    }
    
    all_results = []
    
    # Iterate through all folds
    for fold_dir in sorted(cv_base.glob("cv_fold_*")):
        fold_num = fold_dir.name.split("_")[-1]
        
        print(f"Processing {fold_dir.name}...")
        
        # Iterate through all scenarios in this fold
        for scenario_dir in sorted(fold_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
                
            scenario_key = scenario_dir.name
            scenario_name = scenario_names.get(scenario_key, scenario_key)
            
            # Look for all_summary.csv in this scenario
            summary_file = scenario_dir / "all_summary.csv"
            if not summary_file.exists():
                print(f"  Warning: No all_summary.csv found in {scenario_dir}")
                continue
                
            try:
                # Read the summary CSV
                df = pd.read_csv(summary_file)
                
                # Add fold and scenario information
                df['fold'] = int(fold_num)
                df['scenario'] = scenario_name
                df['scenario_key'] = scenario_key
                
                # Reorder columns to put identifiers first
                cols = ['fold', 'scenario', 'scenario_key', 'exp'] + [c for c in df.columns if c not in ['fold', 'scenario', 'scenario_key', 'exp']]
                df = df[cols]
                
                all_results.append(df)
                print(f"  ✓ {scenario_name}: {len(df)} experiments")
                
            except Exception as e:
                print(f"  Error reading {summary_file}: {e}")
    
    if not all_results:
        print("No results found!")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by fold, then scenario for better organization
    combined_df = combined_df.sort_values(['fold', 'scenario_key', 'exp'])
    
    # Save to CSV
    output_file = "cv_results_aggregated.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Results aggregated successfully!")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Total records: {len(combined_df)}")
    print(f"✓ Folds processed: {combined_df['fold'].nunique()}")
    print(f"✓ Scenarios: {combined_df['scenario'].nunique()}")
    print(f"✓ Unique experiments: {combined_df['exp'].nunique()}")
    
    # Show summary statistics
    print(f"\n📊 Summary by scenario:")
    scenario_summary = combined_df.groupby('scenario').agg({
        'rmse_3d': ['mean', 'std', 'count'],
        'nees': ['mean', 'std']
    }).round(4)
    print(scenario_summary)
    
    return combined_df

if __name__ == "__main__":
    aggregate_cv_results()