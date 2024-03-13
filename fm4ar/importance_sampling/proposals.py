"""
Methods for drawing samples from a proposal distribution.
"""

from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from fm4ar.utils.config import load_config as load_experiment_config
from fm4ar.datasets.theta_scalers import get_theta_scaler
from fm4ar.importance_sampling.config import ImportanceSamplingConfig
from fm4ar.importance_sampling.target_spectrum import load_target_spectrum
from fm4ar.models.build_model import build_model
from fm4ar.nn.flows import create_unconditional_flow_wrapper
from fm4ar.unconditional_flow.config import load_config as load_flow_config


def draw_proposal_samples(
    args: Namespace,
    config: ImportanceSamplingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw samples from the proposal distribution.

    Args:
        args: Command-line arguments.
        config: Configuration for the importance sampling run.

    Returns:
        Tuple of two numpy arrays, '(theta, probs)', containing the
        proposal samples and their respective probabilities.
    """

    # Determine the model type: FMPE / NPE or unconditional flow?
    experiment_config = load_experiment_config(args.experiment_dir)
    model_type = experiment_config["model"]["model_type"]

    # Determine the number of samples that the current job should draw
    n_total = config.draw_proposal_samples.n_samples
    n_for_job = len(np.arange(args.job, n_total, args.n_jobs))
    print(f"Total number of samples to draw:             {n_total:,}")
    print(f"Number of samples to draw for current job:   {n_for_job:,}")
    print()

    # Draw samples either from an FMPE / NPE model...
    if model_type in ["fmpe", "npe"]:

        # Construct the context for the model from the target spectrum
        context = {
            k: torch.from_numpy(v).float() for k, v in
            load_target_spectrum(
                file_path=config.target_spectrum.file_path,
                index=(
                    args.target_index if args.target_index is not None
                    else config.target_spectrum.index
                ),
            ).items()
        }

        # Add the error bars to the context
        n_bins = context["flux"].shape[0]
        sigma = config.likelihood.sigma
        context["error_bars"] = sigma * torch.ones(n_bins).float()

        print("Running for ML model (FMPE / NPE)!\n")
        theta, probs = draw_samples_from_ml_model(
            context=context,
            experiment_dir=args.experiment_dir,
            n_samples=n_for_job,
            chunk_size=config.draw_proposal_samples.chunk_size,
            model_kwargs=config.model_kwargs,
        )

    # ... or from an unconditional flow model
    elif model_type == "unconditional_flow":

        print("Running for unconditional flow model!\n")
        theta, probs = draw_samples_from_unconditional_flow(
            experiment_dir=args.experiment_dir,
            n_samples=n_for_job,
            chunk_size=config.draw_proposal_samples.chunk_size,
        )

    else:  # pragma: no cover
        raise ValueError(f"Unknown model type: {model_type}!")

    return theta, probs


def draw_samples_from_ml_model(
    context: dict[str, torch.Tensor],
    n_samples: int,
    experiment_dir: Path,
    checkpoint_name: str = "model__best.pt",
    chunk_size: int = 1024,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trained ML model (NPE or FMPE) and draw samples from it.

    Args:
        context: Context for the model. This should be a dictionary
            with keys "flux" and "wlen" (and possibly others, such as
            "error_bars").
        n_samples: Number of samples to draw from the model.
        experiment_dir: Path to the experiment directory that holds the
            trained model.
        checkpoint_name: Name of the checkpoint file to load. Defaults
            to "model__best.pt".
        chunk_size: Size of the "chunks" for drawing samples (i.e., the
            number of samples drawn at once). This is used to avoid
            running out of GPU memory.
        model_kwargs: Additional keyword arguments for the model.
            This is useful for specifying the tolerance for FMPE models.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {} if model_kwargs is None else model_kwargs

    # Load the trained model
    print("Loading trained model...", end=" ")
    file_path = experiment_dir / checkpoint_name
    model = build_model(file_path=file_path, device=device)
    model.network.eval()
    print("Done!")

    # Load experiment config and construct a standardizer for the data
    print("Creating standardizer...", end=" ")
    config = load_experiment_config(experiment_dir=experiment_dir)
    theta_scaler = get_theta_scaler(config=config["theta_scaler"])
    print("Done!\n")

    # Determine the chunk sizes: Every chunk should have `chunk_size` samples,
    # except for the last one, which may have fewer samples.
    chunk_sizes = np.diff(np.r_[0: n_samples: chunk_size, n_samples])

    # Draw samples from the model posterior ("proposal distribution")
    print("Drawing samples from the model posterior:", flush=True)
    theta_chunks = []
    probs_chunks = []
    with torch.no_grad():
        for n in tqdm(chunk_sizes, ncols=80):

            # Adjust the size of the context so that the batch size matches
            # the desired chunk size, and move it to the correct device
            chunk_context = {
                k: v.repeat(n, 1).to(device, non_blocking=True)
                for k, v in context.items()
            }

            # Draw samples from the model
            theta_chunk, log_probs_chunk = model.sample_and_log_prob_batch(
                context=chunk_context,
                **model_kwargs,
            )

            # Inverse-transform the samples and convert the log-probabilities
            theta_chunk = theta_scaler.inverse_tensor(theta_chunk.cpu())
            probs_chunk = torch.exp(log_probs_chunk.cpu())
            theta_chunks.append(theta_chunk.cpu())
            probs_chunks.append(probs_chunk.cpu())

    print(flush=True)

    # Combine all chunks into a single array
    theta = torch.cat(theta_chunks, dim=0).numpy()
    probs = torch.cat(probs_chunks, dim=0).numpy().flatten()

    return theta, probs


def draw_samples_from_unconditional_flow(
    experiment_dir: Path,
    n_samples: int,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a trained unconditional flow model and draw samples from it.

    Args:
        experiment_dir: Path to the experiment directory that holds the
            trained model.
        n_samples: Number of samples to draw from the model.
        chunk_size: Size of the "chunks" for drawing samples (i.e., the
            number of samples drawn at once). This is used to avoid
            running out of GPU memory.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the unconditional flow config
    config = load_flow_config(experiment_dir=experiment_dir)

    # Create the scaler for `theta`
    print("Creating standardizer...", end=" ")
    theta_scaler = get_theta_scaler(config=config.theta_scaler)
    print("Done!")

    # Load the model checkpoint
    print("Loading checkpoint...", end=" ")
    checkpoint = torch.load(
        f=experiment_dir / "model__best.pt",
        map_location=torch.device(device),
    )
    print("Done!")

    # Load the unconditional flow model
    print("Loading unconditional flow model...", end=" ")
    model = create_unconditional_flow_wrapper(
        dim_theta=checkpoint["dim_theta"],
        flow_wrapper_config=config.model.flow_wrapper,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Done!")

    # Determine the chunk sizes: Every chunk should have `chunk_size` samples,
    # except for the last one, which may have fewer samples.
    chunk_sizes = np.diff(np.r_[0: n_samples: chunk_size, n_samples])

    # Draw samples from the unconditional flow model
    print("Drawing samples from unconditional flow:", flush=True)
    theta_chunks = []
    probs_chunks = []
    with torch.no_grad():

        # Draw samples in chunks and inverse-transform them
        for n in tqdm(chunk_sizes, ncols=80):
            theta_chunk, logprob_chunk = model.sample_and_log_prob(
                num_samples=n,
            )
            theta_chunk = theta_scaler.inverse_tensor(theta_chunk.cpu())
            probs_chunk = torch.exp(logprob_chunk).cpu()
            theta_chunks.append(theta_chunk)
            probs_chunks.append(probs_chunk)

        # Combine all chunks into a single array
        theta = torch.cat(theta_chunks, dim=0).numpy()
        probs = torch.cat(probs_chunks, dim=0).numpy().flatten()

    print("Done!\n")

    return theta, probs
