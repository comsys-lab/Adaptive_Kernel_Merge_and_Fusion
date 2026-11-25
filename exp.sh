#!/bin/bash
# Build the shared libraries and run every policy at every arrival rate.
# Run from the repository root. Needs data_alexnet/, data_resnet18/, data_vgg16/
# (create them with ./export_weights.py).
set -e

NUM_REQUESTS=${NUM_REQUESTS:-1000}
RATES="95 90 85 80"

for lib in basic_models aras_models merge_fusion; do
    nvcc -Xcompiler -fPIC -shared -o ${lib}.so src/${lib}.cu
done

mkdir -p results
for rate in $RATES; do
    for policy in sequential concurrent aras merge_fusion; do
        echo "=== ${policy} (arrival rate ${rate}%)"
        ./experiments_${policy}.py ${NUM_REQUESTS} ${rate}
        # Result files are opened in append mode: keep one file per run.
        mv results/${policy}.txt results/${policy}_${rate}.txt
    done
done

./parsing.py results/*_*.txt
