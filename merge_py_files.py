#!/usr/bin/env python3
import os
import sys

def collect_py_files(directories, specific_files):
    files = []
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, filenames in os.walk(directory):
                # Skip __pycache__
                if '__pycache__' in dirs:
                    dirs.remove('__pycache__')
                for filename in filenames:
                    if filename.endswith('.py'):
                        files.append(os.path.join(root, filename))
    for file_path in specific_files:
        if os.path.exists(file_path) and file_path.endswith('.py'):
            files.append(file_path)
    return files

def merge_files(files, output_file):
    with open(output_file, 'w') as outfile:
        for file_path in files:
            try:
                with open(file_path, 'r') as infile:
                    content = infile.read()
                # Write filename as header
                outfile.write(f"### {os.path.basename(file_path)}\n")
                outfile.write(content)
                outfile.write("\n\n")  # Separator
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    base_dir = "/home/nick/Thesis"
    directories = [
        os.path.join(base_dir, "swarm_ml"),
        os.path.join(base_dir, "swarm_control"),
        os.path.join(base_dir, "swarm_net"),
        os.path.join(base_dir, "agents")
    ]
    specific_files = [
        os.path.join(base_dir, "swarm_batch_run.py"),
        os.path.join(base_dir, "swarm_eval_table.py"),
        os.path.join(base_dir, "swarm_target_tracking.py")
    ]
    output_file = os.path.join(base_dir, "merged_py_files.txt")
    
    files = collect_py_files(directories, specific_files)
    merge_files(files, output_file)
    print(f"Merged {len(files)} files into {output_file}")