#!/usr/bin/env python3
"""
Create a summary table of the cross-validation results.
"""

import pandas as pd
import numpy as np

def create_summary_table():
    """Create a clean summary table from the aggregated CV results."""
    
    # Read the aggregated results
    df = pd.read_csv("cv_results_aggregated.csv")
    
    # Calculate summary statistics by scenario
    summary = df.groupby('scenario').agg({
        'rmse_3d': ['mean', 'std'],
        'nees': ['mean', 'std'],
        'fold': 'count'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['RMSE_3D_Mean', 'RMSE_3D_Std', 'NEES_Mean', 'NEES_Std', 'N_Folds']
    
    # Reset index to make scenario a column
    summary = summary.reset_index()
    
    # Sort by RMSE performance (best first)
    summary = summary.sort_values('RMSE_3D_Mean')
    
    # Add performance ranking
    summary['RMSE_Rank'] = summary['RMSE_3D_Mean'].rank()
    summary['NEES_Rank'] = summary['NEES_Mean'].rank()
    
    # Calculate confidence intervals (assuming normal distribution)
    summary['RMSE_CI_Lower'] = summary['RMSE_3D_Mean'] - 1.96 * summary['RMSE_3D_Std'] / np.sqrt(summary['N_Folds'])
    summary['RMSE_CI_Upper'] = summary['RMSE_3D_Mean'] + 1.96 * summary['RMSE_3D_Std'] / np.sqrt(summary['N_Folds'])
    
    summary['NEES_CI_Lower'] = summary['NEES_Mean'] - 1.96 * summary['NEES_Std'] / np.sqrt(summary['N_Folds'])
    summary['NEES_CI_Upper'] = summary['NEES_Mean'] + 1.96 * summary['NEES_Std'] / np.sqrt(summary['N_Folds'])
    
    # Round confidence intervals
    summary[['RMSE_CI_Lower', 'RMSE_CI_Upper', 'NEES_CI_Lower', 'NEES_CI_Upper']] = summary[['RMSE_CI_Lower', 'RMSE_CI_Upper', 'NEES_CI_Lower', 'NEES_CI_Upper']].round(4)
    
    # Reorder columns for better presentation
    summary = summary[['scenario', 'RMSE_3D_Mean', 'RMSE_3D_Std', 'RMSE_CI_Lower', 'RMSE_CI_Upper', 'RMSE_Rank',
                      'NEES_Mean', 'NEES_Std', 'NEES_CI_Lower', 'NEES_CI_Upper', 'NEES_Rank', 'N_Folds']]
    
    # Save summary table
    summary.to_csv("cv_results_summary.csv", index=False)
    
    print("📊 Cross-Validation Results Summary")
    print("=" * 50)
    print(f"✓ Summary table saved to: cv_results_summary.csv")
    print(f"✓ Based on {summary['N_Folds'].iloc[0]} cross-validation folds")
    print("")
    
    # Display key findings
    print("🏆 Performance Ranking (by RMSE_3D):")
    for i, row in summary.iterrows():
        print(f"{int(row['RMSE_Rank']):2d}. {row['scenario']:25s} RMSE: {row['RMSE_3D_Mean']:.4f}±{row['RMSE_3D_Std']:.4f}m  NEES: {row['NEES_Mean']:.4f}±{row['NEES_Std']:.4f}")
    
    print("")
    print("🎯 Key Insights:")
    best_rmse = summary.iloc[0]
    worst_rmse = summary.iloc[-1]
    
    improvement = ((worst_rmse['RMSE_3D_Mean'] - best_rmse['RMSE_3D_Mean']) / worst_rmse['RMSE_3D_Mean']) * 100
    print(f"• Best method ({best_rmse['scenario']}) vs Worst ({worst_rmse['scenario']}): {improvement:.1f}% improvement")
    
    # Compare to baseline
    baseline_idx = summary[summary['scenario'] == 'Baseline_Grid'].index
    if len(baseline_idx) > 0:
        baseline = summary.iloc[baseline_idx[0]]
        best = summary.iloc[0]
        if best['scenario'] != 'Baseline_Grid':
            baseline_improvement = ((baseline['RMSE_3D_Mean'] - best['RMSE_3D_Mean']) / baseline['RMSE_3D_Mean']) * 100
            print(f"• Best method vs Baseline: {baseline_improvement:.1f}% improvement")
    
    return summary

if __name__ == "__main__":
    create_summary_table()