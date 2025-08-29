# swarm_eval_table.py
import argparse, glob, os, csv
import pandas as pd

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs_swarm")
    args = p.parse_args()

    rows = []
    for summ in glob.glob(os.path.join(args.root, "*", "summary.csv")):
        exp_dir = os.path.dirname(summ)
        exp_name = os.path.basename(exp_dir)
        df = pd.read_csv(summ)
        d = df.iloc[0].to_dict()
        d["exp"] = exp_name
        rows.append(d)
    out = pd.DataFrame(rows).sort_values("exp")
    out.to_csv(os.path.join(args.root, "all_summary.csv"), index=False)
    print(out)