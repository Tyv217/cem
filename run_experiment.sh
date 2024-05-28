#!/bin/bash
# python experiments/run_experiments_acflow.py -c experiments/configs/afa_configs/mnist_acflow.yaml --debug
# python experiments/run_experiments_rl.py -c experiments/configs/afa_configs/mnist_afa_copy.yaml --debug --project_name=mnist_verify
# python experiments/run_experiments_ac.py -c experiments/configs/afa_configs/mnist_acenergy_cem.yaml --debug --force_cpu
# python experiments/run_experiments_rl.py -c experiments/configs/afa_configs/mnist_adam.yaml --debug --project_name="mnist_acflow"
python experiments/run_experiments_rl.py -c experiments/configs/afa_configs/cub_intcem.yaml --debug --project_name="cub_test"

