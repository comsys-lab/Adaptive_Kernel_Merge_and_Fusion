#!/usr/bin/env python3
"""Summarize a result file: average queuing delay, inference time, and latency.

Usage: ./parsing.py results/merge_fusion.txt [more files ...]

Each line of a result file is `<model>\t<queuing_delay>\t<inference_time>[\t...]`
(seconds). Latency = queuing delay + inference time.
"""
import sys

import numpy as np


def summarize(path):
    qd, inf = [], []
    with open(path) as f:
        for line in f:
            fields = [x.strip() for x in line.split("\t") if x.strip()]
            if len(fields) < 3:
                continue
            qd.append(float(fields[1]))
            inf.append(float(fields[2]))
    if not qd:
        print(f"{path}: no requests")
        return
    qd, inf = np.array(qd), np.array(inf)
    lat = qd + inf
    print(f"{path}")
    print(f"  requests             : {len(qd)}")
    print(f"  avg queuing delay    : {qd.mean():.6f} s")
    print(f"  avg inference time   : {inf.mean():.6f} s")
    print(f"  avg latency          : {lat.mean():.6f} s")
    print(f"  99th-pct latency     : {np.percentile(lat, 99):.6f} s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().splitlines()[2])
    for p in sys.argv[1:]:
        summarize(p)
