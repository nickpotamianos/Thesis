#!/usr/bin/env python3
"""
Per-robot agent stub.
Responsibilities:
 - run local TargetIF + MeasureAdapter + OnlineTuner
 - publish compact messages (mu, P, features)
Transport: to be implemented (UDP/ZeroMQ).
"""
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--udp", default="239.0.0.1:5001")
    args = p.parse_args()
    print(f"[AGENT] robot_node stub for {args.id} on {args.udp}")

if __name__ == "__main__":
    main()

