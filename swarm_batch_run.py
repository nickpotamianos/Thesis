# swarm_batch_run.py
import argparse, subprocess, json, os

def run_one(exp, argslist):
    cmd = ["python", "swarm_target_tracking.py", "--exp", exp] + argslist
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exps", nargs="+", required=True, help="List of experiments")
    p.add_argument("--common", nargs="*", default=[], help="Args to forward to swarm_target_tracking.py")
    args, unknown = p.parse_known_args()
    
    # Combine common args with any additional args
    common_args = args.common + unknown

    for exp in args.exps:
        run_one(exp, common_args)