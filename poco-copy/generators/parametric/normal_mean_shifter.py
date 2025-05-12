import numpy as np
import pandas as pd
from poco.generators.parametric.parametric_base import BaseParametricGenerator


class NormalMeanShifter(BaseParametricGenerator):
    """
    Parametric generator that simulates a normal distribution and shifts the mean
    to simulate an effect size.

    This is useful for controlled simulations when you don't want to rely on real data.
    """

    def __init__(
        self, n_samples: int = 100, baseline_mean: float = 0.0, std: float = 1.0
    ):
        """
        Parameters
        ----------
        n_samples : int
            Number of samples to generate per simulation.

        baseline_mean : float
            Mean under the null (typically 0.0).

        std : float
            Standard deviation of the normal distribution.
        """
        self.n_samples = n_samples
        self.baseline_mean = baseline_mean
        self.std = std

    def generate(self, effect_size: float, seed: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic normal data with mean = baseline_mean + effect_size.

        Parameters
        ----------
        effect_size : float
            The size of the effect to shift the mean by.

        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one column 'y' containing the simulated values.
        """
        rng = np.random.default_rng(seed)
        shifted_mean = self.baseline_mean + effect_size
        values = rng.normal(loc=shifted_mean, scale=self.std, size=self.n_samples)

        return pd.DataFrame({"y": values})
