#!/bin/bash

RUN_PATH=${1:?"Usage: $0 /path/to/model-run"}
BASE_CMD="CUDA_VISIBLE_DEVICES=0 python -m mldft.ofdft.run_density_optimization run_path='$RUN_PATH' n_molecules=100 save_coeff_interval=1 density_optimizer.max_cycle=1000 density_optimizer.scheduler.min_lr=0.003 density_optimizer.scheduler.max_lr=0.009 density_optimizer.scheduler.decay_rate=0.995 density_optimizer.scheduler.min_lr_decay=0.003"

# Loop over offset and slope values
for offset in $(seq 3.5 0.5 5); do
    for slope in $(seq 1 0.5 3); do
        echo "Running with offset=$offset and slope=$slope"
        eval "$BASE_CMD density_optimizer.scheduler.offset=$offset density_optimizer.scheduler.slope=$slope"
    done
done
