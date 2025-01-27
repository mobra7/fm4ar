"""
Utility functions for nested sampling.
"""

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chainconsumer import Chain, ChainConsumer, PlotConfig, Truth

from fm4ar.priors.base import BasePrior
from fm4ar.priors.config import PriorConfig


def get_parameter_masks(
    prior: BasePrior,
    config: PriorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get binary masks indicating which parameters are being inferred,
    which are being marginizalized over, and which are fixed.

    Args:
        prior: The prior object.
        config: The configuration of the prior.

    Returns:
        A tuple of four numpy arrays:
        - infer_mask: A boolean array indicating which parameters are
            being inferred.
        - marginalize_mask: A boolean array indicating which parameters
            are being marginalized over.
        - condition_mask: A boolean array indicating which parameters
            are being conditioned on.
        - condition_values: An array of the values of the conditioned
            parameters.
    """

    # Create empty masks
    infer_mask = np.zeros(len(prior.names), dtype=bool)
    marginalize_mask = np.zeros(len(prior.names), dtype=bool)
    condition_mask = np.zeros(len(prior.names), dtype=bool)
    condition_values = np.full(len(prior.names), np.nan)

    # Loop over the parameters in the order of the prior and create the masks
    for i, name in enumerate(prior.names):

        # Get the action for the parameter
        try:
            action = config.parameters[name]
        except KeyError as e:
            raise KeyError(
                f"Parameter '{name}' not found in the configuration!"
            ) from e

        # Set the masks based on the action
        if action == "infer":
            infer_mask[i] = True
        elif action == "marginalize":
            marginalize_mask[i] = True
        elif "condition" in action:
            condition_mask[i] = True
            condition_values[i] = float(action.split("=")[1])
        else:
            raise ValueError(
                f"Unknown action '{action}' for parameter '{name}'"
            )

    # Return the masks
    return infer_mask, marginalize_mask, condition_mask, condition_values


def create_posterior_plot(
    samples: np.ndarray,
    weights: np.ndarray,
    names: Sequence[str],
    extents: tuple[np.ndarray, np.ndarray] | None,
    ground_truth: np.ndarray,
    file_path: Path,
) -> plt.Figure:
    """
    Create a corner plot of the posterior.
    """

    # Suppress the custom warnings from ChainConsumer
    logging.basicConfig(level=logging.ERROR)

    # Create the corner plot using ChainConsumer and add the ground truth
    c = ChainConsumer()
    c.add_truth(
        Truth(
            location={k: v for k, v in zip(names, ground_truth, strict=True)},
        )
    )

    # Construct data frame with samples. We need to drop the zero-weight
    # samples because the new version of ChainConsumer can't handle them...
    df = pd.DataFrame(
        data=np.column_stack([samples, weights]),
        columns=list(names) + ["weight"],
    )
    df = df[df['weight'] > 0]

    # Add a chain with the actual samples
    c.add_chain(
        Chain(
            samples=df,
            name="posterior",
        )
    )

    # Set the plot limits to match the prior, if desired
    # If the extents are not provided, chainconsumer will automatically
    # determine them from the data.
    if extents is not None:

        # TODO: If we ever need to handle infite supports (e.g., Gaussian
        #  priors), we might need to truncate things to some reasonable range.
        if (  # pragma: no cover
            np.isinf(extents[0]).any()
            or np.isinf(extents[1]).any()
        ):
            raise ValueError("Infinite supports are currently not supported!")

        # Unpack the extents and set them in the plot config
        lower, upper = extents
        c.set_plot_config(
            PlotConfig(
                extents={
                    name: (float(lower[i]), float(upper[i]))
                    for i, name in enumerate(names)
                }
            )
        )

    # Create the plot
    fig = c.plotter.plot()

    # Save the plot
    plt.savefig(file_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    return fig
