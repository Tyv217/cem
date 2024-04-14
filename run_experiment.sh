#!/bin/bash
# python experiments/run_experiments_acflow.py -c experiments/configs/afa_configs/mnist_acflow.yaml --debug
# python experiments/run_experiments_ac.py -c experiments/configs/afa_configs/mnist_acflow_cem.yaml --debug --rerun --project_name="test"
python experiments/run_experiments_ac.py -c experiments/configs/afa_configs/mnist_acenergy_cem.yaml --debug --rerun --project_name="test"
