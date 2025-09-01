#!/usr/bin/env python3
"""
Central logger stub for decentralized messages.
"""
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--udp", default="239.0.0.1:5001")
    p.add_argument("--out", default="out_decentralized")
    args = p.parse_args()
    print(f"[AGENT] logger stub on {args.udp}, saving to {args.out}")

if __name__ == "__main__":
    main()

