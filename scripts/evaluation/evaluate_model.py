"""
Evaluate a trained model (usually on the test set).

This script can either be run directly on a GPU node, or it can be
invoked using the `--start-submission` flag to prepare a submission
file and launch a new evaluation job on the cluster.
"""

import argparse
import time
from pathlib import Path
from typing import Any,  Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fm4ar.datasets import load_dataset
from fm4ar.datasets.scaling import Scaler
from fm4ar.datasets.vasist_2023.prior import LOWER, UPPER
from fm4ar.models.build_model import build_model
from fm4ar.models.continuous.flow_matching import FlowMatching
from fm4ar.models.discrete.normalizing_flow import NormalizingFlow
from fm4ar.utils.config import load_config
from fm4ar.utils.git_utils import get_git_hash
from fm4ar.utils.hashing import get_sha512sum
from fm4ar.utils.htcondor import (
    HTCondorConfig,
    check_if_on_login_node,
    condor_submit_bid,
    create_submission_file,
)


def get_cli_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device on which to run everything.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory with config and checkpoint.",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default="filtered.hdf",
        help="Name of the HDF file to load (combines with --which).",
    )
    parser.add_argument(
        "--get-logprob-samples",
        type=bool,
        default=False,
        help="Whether to compute the log probability of the samples.",
    )
    parser.add_argument(
        "--get-logprob-theta",
        type=bool,
        default=True,
        help="Whether to compute the log probability of the theta.",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=0,
        help="Job number for parallel processing; must be in [0, n_jobs).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dopri5",  # dopri8 is more stable, but MUCH slower
        help="Method for ODE solver (only needed for flow matching).",
    )
    parser.add_argument(
        "--n-dataset-samples",
        type=int,
        default=1_000,
        help="Number of spectra for which to draw posterior samples.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel.",
    )
    parser.add_argument(
        "--n-posterior-samples",
        type=int,
        default=10_000,
        help="Number of samples to draw from posterior.",
    )
    parser.add_argument(
        "--start-submission",
        action="store_true",
        help=(
            "If this flag is used, the script will prepare the HTCondor "
            "submission file and launch a new job (but not actually run the "
            "evaluation itself)."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for ODE solver (only needed for flow matching).",
    )
    parser.add_argument(
        "--which",
        type=str,
        default="test",
        help="Which dataset to use for evaluation purposes.",
    )
    args = parser.parse_args()

    return args


def get_logprob_of_theta(
    model: FlowMatching | NormalizingFlow,
    theta: torch.Tensor,
    x: torch.Tensor,
    device: Literal["cpu", "cuda"],
    **model_kwargs: Any,
) -> float:
    """
    Compute the log probability of `theta` given `x`.
    """

    model.model.eval()

    # AMP only makes sense for `FlowMatching` models
    use_amp = isinstance(model, FlowMatching)

    # For some reason, this does occasionally crash with an assertion
    # error: "AssertionError: underflow in dt nan"
    # Idea: Use `dopri8` instead of `dopri5` as the solver?
    with (
        torch.autocast(device_type=device, enabled=use_amp),
        torch.no_grad(),
    ):
        try:
            logprob_theta = (
                model.log_prob_batch(
                    theta.to(device), x.to(device), **model_kwargs
                )
                .cpu()
                .numpy()
                .squeeze()
            )
        except AssertionError as e:
            print(e)
            logprob_theta = np.nan

    return float(logprob_theta)


def get_samples(
    model: FlowMatching | NormalizingFlow,
    x: torch.Tensor,
    n_samples: int,
    theta_scaler: Scaler,
    batch_size: int = 512,
    device: Literal["cpu", "cuda"] = "cuda",
    get_logprob_samples: bool = False,
    **model_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw samples from posterior (with or without log probability).
    """

    model.model.eval()

    # AMP only makes sense for `FlowMatching` models
    use_amp = isinstance(model, FlowMatching)

    with (
        torch.autocast(device_type=device, enabled=use_amp),
        torch.no_grad(),
    ):

        # Prepare input for model
        x = x.tile(batch_size, 1, 1).to(device)

        # Draw samples in a batch-wise fashion (this is generally faster).
        # This also allows us to discard samples that are outside the prior
        # and still get the correct number of samples in the end.
        all_samples = np.empty(shape=(0, len(LOWER)))
        all_logprob = np.empty(shape=(0, ))
        while len(all_samples) < n_samples:

            # Draw samples from the posterior and compute log probability
            # For flow matching, this can sometimes give an AssertionError,
            # which (probably) means the ODE was too stiff. In this case, we
            # just try again with a different random batch. (Alternatively,
            # one can try to use `dopri8` instead of `dopri5` as the solver,
            # but this is MUCH slower.)
            try:
                if get_logprob_samples:
                    (
                        samples_torch,
                        logprob_torch,
                    ) = model.sample_and_log_prob_batch(x, **model_kwargs)
                    logprob_numpy = logprob_torch.cpu().numpy()
                else:
                    samples_torch = model.sample_batch(x, **model_kwargs)
                    logprob_numpy = np.full(batch_size, np.nan)

            # Skip this batch if we get an assertion error due to a stiff ODE
            except AssertionError as e:
                if "underflow in dt nan" in str(e):
                    print("\nGot AssertionError! Trying another batch...\n")
                    continue
                else:
                    raise e

            # Map samples back to original units
            samples_torch = theta_scaler.inverse(samples_torch.cpu())
            samples_numpy = samples_torch.numpy().squeeze()

            # Discard samples that are outside the prior range
            idx = np.all(
                np.logical_and(
                    samples_numpy >= np.array(LOWER),
                    samples_numpy <= np.array(UPPER),
                ),
                axis=1,
            )
            samples_numpy = samples_numpy[idx]
            logprob_numpy = logprob_numpy[idx]

            # Store accepted samples
            all_samples = np.concatenate([all_samples, samples_numpy], axis=0)
            all_logprob = np.concatenate([all_logprob, logprob_numpy], axis=0)

    # Make sure we have the correct number of samples
    all_samples = all_samples[:n_samples]
    all_logprob = all_logprob[:n_samples]

    return all_samples, all_logprob


def prepare_submission_file_and_launch_job(args: argparse.Namespace) -> None:
    """
    Create a submission file and launch a new job on the cluster, which
    will run *without* the `--start-submission` flag. This job will then
    run the actual evaluation.
    """

    print("Preparing submission file...", end=" ")

    # Collect arguments for the job: Start with the path to this script,
    # then add all the arguments that we got from the command line
    job_arguments = [
        Path(__file__).resolve().as_posix(),
        f"--batch-size {args.batch_size}",
        f"--device {args.device}",
        f"--experiment-dir {args.experiment_dir}",
        f"--file-name {args.file_name}",
        f"--get-logprob-samples {args.get_logprob_samples}",
        f"--get-logprob-theta {args.get_logprob_theta}",
        "--job $(Process)",
        f"--method {args.method}",
        f"--n-dataset-samples {args.n_dataset_samples}",
        f"--n-jobs {args.n_jobs}",
        f"--n-posterior-samples {args.n_posterior_samples}",
        f"--tolerance {args.tolerance}",
        f"--which {args.which}",
    ]

    # Prepare the condor settings. The evaluation basically happens only on
    # the GPU, so we don't need a lot of CPUs, and just enough memory to load
    # the dataset and hold the results.
    condor_settings = HTCondorConfig(
        num_cpus=1,
        memory_cpus=50_000,
        num_gpus=1,
        arguments=job_arguments,
        log_file_name=f"evaluate_on_{args.which}.$(Process)",
        queue=args.n_jobs,
        bid=25,
    )

    # Create submission file and submit job
    file_path = create_submission_file(
        condor_settings=condor_settings,
        experiment_dir=args.experiment_dir,
        file_name=f"evaluate_on_{args.which}.sub",
    )

    print("Done!\n")

    condor_submit_bid(bid=condor_settings.bid, file_path=file_path)


def run_evaluation(args: argparse.Namespace) -> None:
    """
    Run the actual evaluation (either on test or training data).
    """

    script_start = time.time()

    # Load the experiment configuration
    config = load_config(experiment_dir=args.experiment_dir)

    # Update the experiment configuration
    config["data"]["which"] = args.which
    config["data"]["file_name"] = args.file_name
    config["data"]["add_noise_to_x"] = False
    config["data"]["n_samples"] = int(args.n_dataset_samples)

    # Load the dataset
    # Note 1: The test set should load the flux with a fixed noise realization
    # for each sample. There is currently no way to specify the key that needs
    # to be loaded from the HDF file, so we need to manually make sure that
    # the test file contains something like `flux` = `raw_flux` + `'noise`.
    # Note 2: We need to select the subset of the dataset for the current job,
    # which is given by `args.job` and `args.n_jobs`.
    print("Loading dataset...", end=" ")
    dataset = load_dataset(config)
    dataset.flux = dataset.flux[args.job :: args.n_jobs, :]
    dataset.theta = dataset.theta[args.job :: args.n_jobs, :]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    print("Done!")

    # Load the model
    print("Loading model...", end=" ")
    file_path = args.experiment_dir / "model__best.pt"
    model_hash = get_sha512sum(file_path=file_path)
    model = build_model(file_path=file_path, device=args.device)
    model.model.eval()
    print("Done!\n")

    # Define model-specific keyword arguments
    if isinstance(model, FlowMatching):
        model_kwargs = {
            "tolerance": args.tolerance,
            "method": args.method,
        }
    else:
        model_kwargs = {}

    # Prepare the values that we want to save later
    list_of_thetas: list[np.ndarray] = []
    list_of_samples: list[np.ndarray] = []
    list_of_logprob_thetas: list[float] = []
    list_of_logprob_samples: list[np.ndarray] = []

    # Evaluate the model
    print("Evaluating model:")
    for theta, x in tqdm(dataloader, ncols=80):

        # Compute log probability of theta
        if args.get_logprob_theta:
            logprob_theta = get_logprob_of_theta(
                model=model,
                theta=theta,
                x=x,
                device=args.device,
                **model_kwargs,
            )
            list_of_logprob_thetas.append(logprob_theta)

        # Draw samples from posterior and store the result
        samples, logprob_samples = get_samples(
            model=model,
            x=x,
            n_samples=args.n_posterior_samples,
            theta_scaler=dataset.theta_scaler,
            device=args.device,
            batch_size=args.batch_size,
            get_logprob_samples=args.get_logprob_samples,
            **model_kwargs,
        )
        list_of_samples.append(samples)
        list_of_logprob_samples.append(logprob_samples)

        # Store theta (in original units)
        theta = dataset.theta_scaler.inverse(theta)
        list_of_thetas.append(theta.numpy())

    print()

    # Convert lists to numpy arrays
    thetas = np.array(list_of_thetas).squeeze(1)
    samples = np.array(list_of_samples)
    logprob_thetas = np.array(list_of_logprob_thetas)
    logprob_samples = np.array(list_of_logprob_samples)

    # Prepare the results directory
    evaluation_dir = args.experiment_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)

    # Save the results to an HDF file
    print("Saving results...", end=" ")
    file_path = evaluation_dir / f"results_{args.which}__{args.job:03d}.hdf"
    with h5py.File(file_path, "w") as f:
        f.attrs["model_hash"] = model_hash
        f.attrs["git_hash"] = get_git_hash()
        f.create_dataset(name="theta", data=thetas)
        f.create_dataset(name="samples", data=samples)
        if args.get_logprob_theta:
            f.create_dataset(name="logprob_theta", data=logprob_thetas)
        if args.get_logprob_samples:
            f.create_dataset(name="logprob_samples", data=logprob_samples)
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds!\n")


if __name__ == "__main__":

    print("\nEVALUATE MODEL\n")

    # Parse arguments and load experiment configuration
    args = get_cli_arguments()

    # Make sure we don't try to run the actual evaluation on the login node
    check_if_on_login_node(start_submission=args.start_submission)

    # Check if we need to prepare a submission file and launch a new job, or
    # if we run the actual evaluation
    if args.start_submission:
        prepare_submission_file_and_launch_job(args=args)
    else:
        run_evaluation(args=args)
