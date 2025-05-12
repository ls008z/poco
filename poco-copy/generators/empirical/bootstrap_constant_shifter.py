"""
poco.generators.bootstrap_constant_shifter

An empirical generator that creates bootstrap resamples from real data
and simulates constant treatment effects by shifting the outcome variable.
"""

import numpy as np
import pandas as pd
from poco.generators.empirical.empirical_base import BaseEmpiricalGenerator


class BootstrapConstantShifter(BaseEmpiricalGenerator):
    """
    Empirical generator that bootstraps from observed data and injects
    a constant additive effect into a specified outcome column.

    Each call to `generate()` returns a new resampled dataset with the
    effect size applied. Useful for simulating treatment effects in power
    and coverage studies.
    """

    def __init__(self, base_data: pd.DataFrame, outcome_col: str):
        """
        Initialize the generator with base data and the outcome column.

        Parameters
        ----------
        base_data : pd.DataFrame
            The original dataset to resample from.

        outcome_col : str
            The column in the dataset to which the effect size will be added.
        """
        super().__init__(base_data)
        self.outcome_col = outcome_col

    def generate(self, effect_size: float, seed: int, **kwargs) -> pd.DataFrame:
        """
        Generate a bootstrap sample with a constant shift applied to the outcome column.

        Parameters
        ----------
        effect_size : float
            The amount to shift the outcome column by. This simulates a constant treatment effect.

        seed : int
            Random seed for reproducibility.

        kwargs : dict
            Additional arguments (currently unused).

        Returns
        -------
        pd.DataFrame
            A new dataset with the same structure as `base_data`, bootstrapped and modified.
        """
        rng = np.random.default_rng(seed)
        resampled = self.base_data.sample(
            n=len(self.base_data), replace=True, random_state=rng.integers(1e9)
        ).copy()

        resampled[self.outcome_col] += effect_size
        return resampled.reset_index(drop=True)


if __name__ == "__main__":
    # Step 1: Create some baseline data (e.g., observed from a control group)
    df = pd.DataFrame(
        {"y": np.random.normal(loc=0.0, scale=1.0, size=100)}  # Outcome with mean 0
    )

    # Step 2: Instantiate the generator
    generator = BootstrapConstantShifter(base_data=df, outcome_col="y")

    # Step 3: Generate a new dataset with a simulated effect size
    simulated = generator.generate(effect_size=1.5, seed=123)

    # Step 4: Show a comparison of means
    print("Original mean:", df["y"].mean())
    print("Simulated mean:", simulated["y"].mean())

    # Optionally: look at the first few rows
    print(simulated.head())
