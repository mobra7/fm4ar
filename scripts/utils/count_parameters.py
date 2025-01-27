"""
Instantiate a model from a config file and count the number of
parameters without starting a training run.
"""

from argparse import ArgumentParser
from pathlib import Path
from time import time

import torch

from fm4ar.torchutils.general import get_number_of_parameters
from fm4ar.training.preparation import prepare_new
from fm4ar.utils.config import load_config

if __name__ == "__main__":

    script_start = time()
    print("\nCOUNT PARAMETERS OF MODEL\n")

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory with config.yaml.",
    )
    args = parser.parse_args()

    print("Experiment directory:", flush=True)
    print(f"{args.experiment_dir.resolve()}\n", flush=True)

    # Load config and update local settings to ensure they work on macOS.
    # Also, load only the smaller test set to get the correct theta_dim.
    config = load_config(args.experiment_dir)
    config["dataset"]["n_train_samples"] = 10
    config["dataset"]["n_valid_samples"] = 0
    config["local"]["device"] = "cpu"
    config["local"]["n_workers"] = 0
    if "wandb" in config["local"]:
        del config["local"]["wandb"]

    # Load data and build model (needed to infer theta_dim and context_dim)
    model, dataset = prepare_new(
        experiment_dir=args.experiment_dir,
        config=config,
    )
    print("\n")

    network: torch.nn.Module
    for name, network in (
        ("total", model.network),
        ("context embedding net", model.network.context_embedding_net),
    ):
        n_trainable = get_number_of_parameters(network, (True,))
        n_fixed = get_number_of_parameters(network, (False,))
        n_total = n_trainable + n_fixed
        print(f"Number of {name} parameters:", flush=True)
        print(f"n_trainable: {n_trainable:,}", flush=True)
        print(f"n_fixed:     {n_fixed:,}", flush=True)
        print(f"n_total:     {n_total:,}\n", flush=True)

    print(f"\nThis took {time() - script_start:.2f} seconds!\n", flush=True)
