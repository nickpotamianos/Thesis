#!/usr/bin/env python3
"""
Diagnose the actual structure of fusion snapshots to understand data issues.
"""

import json
import numpy as np
from collections import Counter, defaultdict

def diagnose_fusion_file(jsonl_path):
    """Examine actual structure of fusion snapshots"""
    
    print(f"Examining: {jsonl_path}\n")
    print("="*70)
    
    # Collect statistics
    line_count = 0
    field_counts = Counter()
    x_lengths = Counter()
    exp_fields = Counter()  # Track which field name is used for experiment
    sample_records = []
    problem_lines = []
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            line_count += 1
            
            try:
                data = json.loads(line.strip())
                
                # Track field names
                for field in data.keys():
                    field_counts[field] += 1
                
                # Check experiment field name variants
                if 'exp' in data:
                    exp_fields['exp'] += 1
                if 'exp_id' in data:
                    exp_fields['exp_id'] += 1
                if 'experiment' in data:
                    exp_fields['experiment'] += 1
                    
                # Check X field
                if 'X' in data:
                    X = data['X']
                    if isinstance(X, list):
                        x_lengths[len(X)] += 1
                        
                        # Check if problematic length
                        if len(X) % 11 != 0 and len(X) != 0:
                            problem_lines.append({
                                'line': line_num,
                                'X_length': len(X),
                                'exp': data.get('exp', data.get('exp_id', 'unknown')),
                                'keys': list(data.keys())
                            })
                    else:
                        x_lengths[f"not_list:{type(X).__name__}"] += 1
                
                # Save first few records as examples
                if len(sample_records) < 3:
                    sample_records.append(data)
                    
            except json.JSONDecodeError as e:
                print(f"JSON error on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                continue
    
    # Report findings
    print(f"Total lines processed: {line_count:,}")
    print(f"\nField frequencies (what fields appear in the records):")
    print("-"*40)
    for field, count in field_counts.most_common():
        pct = 100 * count / line_count
        print(f"  {field:20} {count:7,} ({pct:5.1f}%)")
    
    print(f"\nExperiment field name variants found:")
    print("-"*40)
    for field, count in exp_fields.most_common():
        print(f"  '{field}': {count:,} occurrences")
    
    print(f"\nX field lengths distribution:")
    print("-"*40)
    for length, count in sorted(x_lengths.items(), key=lambda x: str(x[0])):
        if isinstance(length, int):
            n_nodes = length // 11
            remainder = length % 11
            if remainder == 0:
                print(f"  Length {length:4}: {count:6,} snapshots ({n_nodes} nodes, valid)")
            else:
                print(f"  Length {length:4}: {count:6,} snapshots (INVALID: {n_nodes} nodes + {remainder} extra)")
        else:
            print(f"  {length}: {count:,}")
    
    # Show sample records
    print(f"\nSample records structure:")
    print("="*70)
    for i, record in enumerate(sample_records, 1):
        print(f"\nRecord {i}:")
        print(f"  Keys: {list(record.keys())}")
        if 'X' in record:
            X = record['X']
            if isinstance(X, list):
                print(f"  X length: {len(X)}")
                if len(X) > 0:
                    print(f"  X first 5 values: {X[:5]}")
            else:
                print(f"  X type: {type(X)}")
        if 'exp' in record:
            print(f"  exp: {record['exp']}")
        if 'exp_id' in record:
            print(f"  exp_id: {record['exp_id']}")
        if 'order' in record:
            print(f"  order (trackers): {record['order']}")
    
    # Problem summary
    if problem_lines:
        print(f"\n⚠️  PROBLEMATIC SNAPSHOTS (X length not divisible by 11):")
        print("="*70)
        print(f"Found {len(problem_lines)} problematic snapshots")
        
        # Group by X length
        by_length = defaultdict(list)
        for p in problem_lines:
            by_length[p['X_length']].append(p)
        
        for length in sorted(by_length.keys()):
            probs = by_length[length]
            print(f"\nLength {length}: {len(probs)} snapshots")
            # Show first few examples
            for p in probs[:3]:
                print(f"  Line {p['line']}: exp={p['exp']}, keys={p['keys']}")
    
    # Diagnosis
    print(f"\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    
    if 2 in x_lengths and x_lengths[2] > 0:
        print("❌ Many snapshots have X length = 2, which is invalid!")
        print("   This suggests incomplete fusion snapshots where only 1 tracker")
        print("   reported (2 values = partial node features).")
        print("   These snapshots should have been filtered out during generation.")
    
    if 22 in x_lengths:
        print("✓ Found snapshots with X length = 22 (2 trackers × 11 features)")
    
    if 33 in x_lengths:
        print("✓ Found snapshots with X length = 33 (3 trackers × 11 features)")
    
    # Recommend action
    print(f"\nRECOMMENDATION:")
    print("-"*40)
    
    if 2 in x_lengths and x_lengths[2] > 100:
        print("Your fusion snapshots file is corrupted with incomplete snapshots.")
        print("You need to either:")
        print("1. Regenerate the fusion snapshots with proper filtering")
        print("2. Filter out invalid snapshots before training:")
        print("   - Keep only snapshots where len(X) % 11 == 0")
        print("   - Keep only snapshots where len(X) >= 22 (at least 2 trackers)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python diagnose_fusion.py <fusion_snaps.jsonl>")
        sys.exit(1)
    
    diagnose_fusion_file(sys.argv[1])