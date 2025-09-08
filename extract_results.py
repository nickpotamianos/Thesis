#!/usr/bin/env python3
import os
import pandas as pd

def extract_experiment_results():
    """Extract RMSE and NEES from experiment summary files"""
    
    base_dir = "/home/nick/Thesis/outputs_swarm"
    
    # Define experiments and their classification
    experiments = {
        # Training experiments (used for FusionNet training)
        "default_3_random_0": "Training (Stable)",
        "default_3_random2_0": "Training (Stable)", 
        "default_3_random3_0b": "Training (Stable)",
        "default_3_zigzag_0": "Training (Challenging)",
        
        # Validation experiments (held out for testing)
        "default_3_random3_2": "Validation (Challenging)",
        "default_3_zigzag_2": "Validation (Challenging)",
        
        # Additional experiments
        "default_3_random_0b": "Additional (Stable)",
    }
    
    print("| Experiment | Target | RMSE_3D | NEES | Classification |")
    print("|------------|---------|---------|------|----------------|")
    
    results = []
    
    for exp_name, classification in experiments.items():
        exp_dir = os.path.join(base_dir, f"{exp_name}_ifo003")
        summary_file = os.path.join(exp_dir, "summary.csv")
        
        if os.path.exists(summary_file):
            try:
                # Read the summary CSV
                df = pd.read_csv(summary_file)
                if not df.empty:
                    # Get the last row (final results)
                    last_row = df.iloc[-1]
                    rmse_3d = last_row.get('rmse_3d', 'N/A')
                    nees = last_row.get('nees', 'N/A')
                    
                    # Format the values
                    if isinstance(rmse_3d, (int, float)):
                        rmse_3d = f"{rmse_3d:.3f}"
                    if isinstance(nees, (int, float)):
                        nees = f"{nees:.3f}"
                    
                    print(f"| {exp_name} | ifo003 | {rmse_3d} | {nees} | {classification} |")
                    results.append({
                        'experiment': exp_name,
                        'rmse_3d': rmse_3d,
                        'nees': nees,
                        'classification': classification
                    })
                else:
                    print(f"| {exp_name} | ifo003 | Empty CSV | Empty CSV | {classification} |")
            except Exception as e:
                print(f"| {exp_name} | ifo003 | Error: {e} | Error: {e} | {classification} |")
        else:
            print(f"| {exp_name} | ifo003 | No summary.csv | No summary.csv | {classification} |")
    
    print("\n### Summary Statistics:")
    
    # Separate training and validation results
    training_rmse = []
    validation_rmse = []
    
    for result in results:
        if 'Training' in result['classification']:
            try:
                training_rmse.append(float(result['rmse_3d']))
            except:
                pass
        elif 'Validation' in result['classification']:
            try:
                validation_rmse.append(float(result['rmse_3d']))
            except:
                pass
    
    if training_rmse:
        print(f"Training experiments average RMSE_3D: {sum(training_rmse)/len(training_rmse):.3f}m")
    if validation_rmse:
        print(f"Validation experiments average RMSE_3D: {sum(validation_rmse)/len(validation_rmse):.3f}m")
    
    return results

if __name__ == "__main__":
    extract_experiment_results()