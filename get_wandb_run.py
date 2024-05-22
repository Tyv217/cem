import numpy as np
import pandas as pd
import argparse
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_path", "-f", type = str, default = None, help = "full path of run")
    parser.add_argument("--project", "-p", type=str, help="Project of run trying to export")
    parser.add_argument("--name", "-n", type=str, help="Name of run trying to export")
    parser.add_argument("--output", "-o", type=str, default = None, help="Output file name")

    args = parser.parse_args()

    if args.output is None:
        output = args.name or args.full_path
    else:
        output = args.output

    api = wandb.Api()
    if args.full_path is None:
        full_path = "thomasyuan1/" + args.project + "/" + args.run
    else:
        full_path = args.full_path
    run = api.run(full_path)
    history = run.history()
    output_path = "wandb_run_output/" + output + ".csv"
    print(f"Exporting run {full_path} to {output_path}")
    history.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()