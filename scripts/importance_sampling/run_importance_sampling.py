"""
Run importance sampling to get posterior samples for a given spectrum.
This script currently only works for the Vasist-2023 dataset.
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from p_tqdm import p_map
from scipy.stats import gaussian_kde
from tqdm import tqdm

from fm4ar.datasets.standardization import get_standardizer
from fm4ar.datasets.vasist_2023.prior import (
    LOWER,
    UPPER,
    NAMES,
    LABELS,
    SIGMA,
    THETA_0,
)
from fm4ar.datasets.vasist_2023.simulation import Simulator
from fm4ar.models.build_model import build_model
from fm4ar.models.continuous.flow_matching import FlowMatching
from fm4ar.nested_sampling.config import load_config as load_ns_config
from fm4ar.nested_sampling.posteriors import load_posterior
from fm4ar.nested_sampling.plotting import create_posterior_plot
from fm4ar.utils.config import load_config as load_ml_config
from fm4ar.utils.htcondor import (
    CondorSettings,
    create_submission_file,
    condor_submit_bid,
)
from fm4ar.utils.multiproc import get_number_of_available_cores


def get_cli_arguments() -> argparse.Namespace:
    """
    Get the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to the experiment directory.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Number of samples to draw from proposal distribution.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1000,
        choices=[400, 1000],
        help="Resolution R = ∆λ/λ of the spectra (default 1000).",
    )
    parser.add_argument(
        "--start-submission",
        action="store_true",
        help="If True, create a submission file and launch a job.",
    )
    args = parser.parse_args()

    return args


def get_parameter_mask() -> np.ndarray:
    """
    Determine mask that indicates which parameters are free, and which
    parameters are fixed to theta_0.
    """

    # In case of an ML model, the config.yaml should contain a "parameters"
    # key with a list of the indices of the free parameters. If there is no
    # explicit "parameters" key, we assume that all parameters are free.
    if (args.experiment_dir / "model__best.pt").exists():
        ml_config = load_ml_config(args.experiment_dir)
        parameters = ml_config["data"].get("parameters")
        if parameters is None:
            parameter_mask = np.ones(len(NAMES), dtype=bool)
        else:
            parameter_mask = np.array(
                [i in parameters for i in range(len(NAMES))]
            )

    # For nested sampling posteriors, the config.yaml (which has a different
    # structure!) explicitly lists all parameters, and we need to select the
    # ones with `action="infer"` (i.e., the free parameters)
    else:
        ns_config = load_ns_config(args.experiment_dir)
        parameter_mask = np.array(
            [ns_config.parameters[name].action == "infer" for name in NAMES]
        )

    return parameter_mask


def process_theta_i(
    theta_i: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Returns the (raw) weight, likelihood, prior, and model probability.
    """

    # NOTE: We do not pass the `simulator`, `x_0`, `context`, ... as arguments
    # to this function because it seems like in that case, the multiprocessing
    # parallelization does not work properly (there is only ever one process
    # at a time doing work). This function can therefore only be called when
    # the outer scope provides these variables!

    # Compute the prior: Since we are using a box-uniform prior, we only need
    # to check if the parameters are within the bounds
    prior = float(
        np.prod(
            [
                float(LOWER[i] <= theta_i[i] <= UPPER[i])
                for i in range(len(theta_i))
            ]
        )
    )

    # If the prior is 0, we can skip the rest (because the weight will be 0)
    if prior == 0:
        return np.full(n_bins, np.nan), 0.0, 0.0

    # Simulate the spectrum that belongs to theta_i
    # If the simulation fails, we simply set the weight to 0.
    result = simulator(theta_i)
    if result is None:
        return np.full(n_bins, np.nan), 0.0, 0.0
    else:
        _, x_i = result

    # Compute the likelihood
    likelihood = float(
        np.exp(float(-0.5 * np.sum(((x_i - x_0) / SIGMA) ** 2)))
    )

    return x_i, likelihood, prior


def handle_nested_sampling_posterior() -> tuple[np.ndarray, np.ndarray]:
    """
    Load posterior samples from nested sampling, apply a KDE, and draw
    samples with corresponding probabilities from the KDE.

    Note: This function accesses values from an outer scope! (This is
    ugly, but in case of the `simulator`, there seems to be no other
    way that works with multiprocessing?)
    """

    # Load the nested sampling posterior
    print("Loading nested sampling posterior...", end=" ")
    ns_samples, ns_weights = load_posterior(experiment_dir=args.experiment_dir)
    print("Done!\n")

    # Fit the posterior with a Gaussian KDE
    print("Fitting nested sampling posterior with KDE...", end=" ")
    kde = gaussian_kde(
        dataset=ns_samples.T,
        weights=ns_weights,
        bw_method=0.1,
    )
    print("Done!")

    # Draw samples from the model posterior ("proposal distribution") and
    # compute probabilities under the model (i.e., the KDE)
    # TODO: Maybe this should also be done in a chunked fashion?
    print("Drawing samples from KDE...", end=" ", flush=True)
    kde_samples = kde.resample(size=args.n_samples).T
    probs = kde.pdf(kde_samples.T)
    theta = np.array(args.n_samples * [THETA_0])
    theta[:, parameter_mask] = kde_samples
    print("Done!\n")

    return theta, probs


def handle_trained_ml_model() -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trained ML model and draw samples from it.

    Note: This function accesses values from an outer scope! (This is
    ugly, but in case of the `simulator`, there seems to be no other
    way that works with multiprocessing?)
    """

    # Construct uncertainties
    noise_level = SIGMA * np.ones_like(x_0)

    # Construct context
    context = (
        torch.stack(
            [
                torch.from_numpy(x_0),
                torch.from_numpy(wlen),
                torch.from_numpy(noise_level),
            ],
            dim=1,
        )
        .float()
        .unsqueeze(0)  # Add batch dimension
    )

    # Load the trained model
    print("Loading trained model...", end=" ")
    file_path = args.experiment_dir / "model__best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(file_path=file_path, device=device)
    model.model.eval()
    print("Done!")

    # Load experiment config and construct a standardizer for the data
    print("Loading standardizer...", end=" ")
    config = load_ml_config(args.experiment_dir)
    standardizer = get_standardizer(config=config)
    print("Done!\n")

    # Define additional keywords for the model
    if isinstance(model, FlowMatching):
        model_kwargs = dict(tolerance=1e-3)
    else:
        model_kwargs = dict()

    # Draw samples from the model posterior ("proposal distribution").
    # We do this in a chunked fashion to avoid running out of GPU memory.
    print("Drawing samples from the model posterior:", flush=True)
    theta_chunks = []
    probs_chunks = []
    chunk_sizes = np.diff(np.r_[0 : args.n_samples : 1000, args.n_samples])
    for chunk_size in tqdm(chunk_sizes, ncols=80):
        with torch.no_grad():
            theta_chunk, log_probs_chunk = model.sample_and_log_prob_batch(
                context=context.repeat(chunk_size, 1, 1).to(device),
                **model_kwargs,  # type: ignore
            )
        theta_chunk = standardizer.inverse_theta(theta_chunk.cpu())
        probs_chunk = torch.exp(log_probs_chunk.cpu())
        theta_chunks.append(theta_chunk.cpu())
        probs_chunks.append(probs_chunk.cpu())
    print(flush=True)

    # Use the parameter mask to combine the sampled theta values with theta_0
    theta_raw = torch.cat(theta_chunks, dim=0).numpy()
    theta = np.repeat(THETA_0[np.newaxis, :], args.n_samples, axis=0)
    theta[:, parameter_mask] = theta_raw

    # Combine the probability chunks into a single numpy array
    probs = torch.cat(probs_chunks, dim=0).numpy().flatten()

    return theta, probs


if __name__ == "__main__":

    script_start = time.time()
    print("\nRUN IMPORTANCE SAMPLING\n")

    # Get the command line arguments
    args = get_cli_arguments()

    # -------------------------------------------------------------------------
    # Either prepare a submission file and launch a job...
    # -------------------------------------------------------------------------

    if args.start_submission:

        # We only need to request a GPU if we are using a trained ML model
        num_gpus = int((args.experiment_dir / "model__best.pt").exists())

        # Collect arguments that we need to pass to the actual job
        arguments = [
            Path(__file__).resolve().as_posix(),
            f"--experiment-dir {args.experiment_dir}",
            f"--n-samples {args.n_samples}",
            f"--resolution {args.resolution}",
        ]

        # Create a submission file for the importance sampling job
        condor_settings = CondorSettings(
            num_cpus=96,
            memory_cpus=65_000,
            num_gpus=num_gpus,
            memory_gpus=15_000,
            arguments=arguments,
            log_file_name="importance_sampling",
            bid=25,
        )
        file_path = create_submission_file(
            condor_settings=condor_settings,
            experiment_dir=args.experiment_dir,
            file_name="importance_sampling.sub",
        )

        # Submit the job to HTCondor
        condor_submit_bid(file_path=file_path, bid=condor_settings.bid)

        # Exit early
        sys.exit(0)

    # -------------------------------------------------------------------------
    # ...or actually run the importance sampling
    # -------------------------------------------------------------------------

    # Set up simulator and compute target spectrum
    print("Simulating target spectrum...", end=" ")
    simulator = Simulator(noisy=False, R=args.resolution)
    if (result := simulator(THETA_0)) is None:
        raise RuntimeError("Simulation of target spectrum failed!")
    else:
        wlen, x_0 = result
    n_bins = len(wlen)
    print("Done!\n")

    # Get the mask that indicates which parameters are free.
    # All other parameters are fixed to theta_0 (required for simulation).
    parameter_mask = get_parameter_mask()

    # Check if the experiment directory we got contains a trained model.
    # In this case, we load the trained model and sample from it.
    if (args.experiment_dir / "model__best.pt").exists():
        print("Found trained ML model!\n")
        theta, probs = handle_trained_ml_model()

    # Otherwise, we assume that the experiment directory contains a nested
    # sampling posterior. In this case, we load the nested sampling posterior,
    # fit a KDE, and sample from it / use it compute probabilities.
    else:
        print("No ML model found, assuming nested sampling posterior!\n")
        theta, probs = handle_nested_sampling_posterior()

    # For each sample, compute the raw weight, likelihood, prior, and prob.
    print("Computing weights for importance sampling:", flush=True)
    num_cpus = get_number_of_available_cores()
    results = p_map(process_theta_i, theta, num_cpus=num_cpus, ncols=80)
    print()

    # Unpack the results from the parallel map
    x, likelihoods, priors = zip(*results, strict=True)
    x = np.array(x)
    likelihoods = np.array(likelihoods).flatten()
    priors = np.array(priors).flatten()

    # Drop everything that has a prior of 0 (i.e., is outside the bounds)
    mask = priors > 0
    theta = theta[mask]
    probs = probs[mask]
    x = x[mask]
    likelihoods = likelihoods[mask]
    priors = priors[mask]
    n = len(theta)
    print(f"Dropped {np.sum(~mask):,} samples where prior=0!")
    print(f"Remaining samples: {n:,} ({100 * n / args.n_samples:.2f}%)\n")

    # Compute the importance sampling weights (raw and normalized)
    raw_is_weights = likelihoods * priors / probs
    is_weights = raw_is_weights * len(raw_is_weights) / np.sum(raw_is_weights)
    print("Min weight:", np.min(is_weights))
    print("Max weight:", np.max(is_weights))
    print()

    # Compute the effective sample size and sample efficiency
    n_eff = np.sum(is_weights) ** 2 / np.sum(is_weights**2)
    sample_efficiency = n_eff / len(is_weights)
    print(f"Effective sample size: {n_eff:.2f}")
    print(f"Sample efficiency:     {100 * sample_efficiency:.2f}%\n")

    # Save the results
    print("Saving results...", end=" ")
    dtype = np.float32
    file_path = args.experiment_dir / "importance_sampling_results.hdf"
    with h5py.File(file_path, "w") as f:
        f.create_dataset(name="parameter_mask", data=parameter_mask)
        f.create_dataset(name="theta_0", data=THETA_0, dtype=dtype)
        f.create_dataset(name="x_0", data=x_0, dtype=dtype)
        f.create_dataset(name="theta", data=theta, dtype=dtype)
        f.create_dataset(name="probs", data=probs, dtype=dtype)
        f.create_dataset(name="x", data=x, dtype=dtype)
        f.create_dataset(name="likelihoods", data=likelihoods, dtype=dtype)
        f.create_dataset(name="raw_weights", data=raw_is_weights, dtype=dtype)
        f.create_dataset(name="priors", data=priors, dtype=dtype)
        f.create_dataset(name="weights", data=is_weights, dtype=dtype)
    print("Done!")

    # Create corner plot comparing the results
    print("Creating posterior plot...", end=" ")
    create_posterior_plot(
        samples=theta,
        weights=is_weights,
        parameter_mask=parameter_mask,
        names=np.array(LABELS)[parameter_mask].tolist(),
        ground_truth=THETA_0[parameter_mask],
        sample_efficiency=sample_efficiency,
        experiment_dir=args.experiment_dir.resolve(),
    )
    print("Done!")

    print(f"\nThis took {time.time() - script_start:.2f} seconds.\n")