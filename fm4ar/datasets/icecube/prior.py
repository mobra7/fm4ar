"""
Define methods to draw a sample from the prior.

This implementation is based on the code from Vasist et al. (2023):
https://github.com/MalAstronomy/sbi-ear
"""

import numpy as np
from scipy.stats import uniform

from fm4ar.priors.base import BasePrior

# Define prior bounds
LOWER: tuple[float]
UPPER: tuple[float]
NAMES: tuple[str]
LABELS: tuple[str]
LOWER, UPPER, NAMES, LABELS = zip(
    *[
        [0., np.pi, "zenith", r"$\phi$"],
        [0., 2*np.pi, "azimuth", r"$\theta$"],
    ],
    strict=True,
)



class Prior(BasePrior):
    """
    Box uniform prior over atmospheric parameters.
    See Table 1 in Vasist et al. (2023).
    """

    def __init__(self, random_seed: int = 42) -> None:
        """
        Initialize class instance.

        Args:
            random_seed: Random seed to use for reproducibility.
        """

        super().__init__(random_seed=random_seed)

        # Store names and labels for the parameters
        self.names = NAMES
        self.labels = LABELS

        # Store prior bounds as arrays
        self.lower = np.array(LOWER)
        self.upper = np.array(UPPER)

        # Construct the prior distribution.
        # Quote from scipy docs: "In the standard form, the distribution is
        # uniform on [0, 1]. Using the parameters loc and scale, one obtains
        # the uniform distribution on [loc, loc + scale]."
        self.distribution = uniform(
            loc=self.lower,
            scale=self.upper - self.lower,
        )
