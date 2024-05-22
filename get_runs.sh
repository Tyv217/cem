#!/bin/bash
source venv/bin/activate
paths=("thomasyuan1/mnist_acflow/exhukt6z" "thomasyuan1/mnist_acflow/fv92apr0" "thomasyuan1/mnist_acflow/50m1xlql")
outputs=("mnist_acflow_split_0" "mnist_acflow_split_1" "mnist_acflow_split_2")

for i in "${!paths[@]}"
do
    python get_wandb_run.py -f="${paths[$i]}" -o="${outputs[$i]}"
done