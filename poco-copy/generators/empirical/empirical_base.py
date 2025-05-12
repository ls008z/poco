"""
poco.generators.empirical_base

Defines the base class for all empirical data generators in the POCO framework.
Empirical generators use real data and modify it via resampling, permutation,
or other methods to simulate datasets under different effect sizes.
"""

from abc import abstractmethod
from poco.generators.base import BaseDataGenerator


class BaseEmpiricalGenerator(BaseDataGenerator):
    """
    Abstract base class for empirical data generators in POCO.

    Empirical generators start with observed data and simulate datasets by modifying
    that data in structured ways (e.g., bootstrap, jackknife, permutation).
    """

    def __init__(self, base_data):
        """
        Initialize the empirical generator.

        Parameters
        ----------
        base_data : Any
            The original dataset to resample or modify (typically a DataFrame).
        """
        self.base_data = base_data

    @abstractmethod
    def generate(self, effect_size: float, seed: int, **kwargs):
        """
        Abstract method to generate a simulated dataset with a specified effect size
        and random seed for reproducibility.

        Parameters
        ----------
        effect_size : float
            Desired effect to inject into the dataset (e.g., mean shift).

        seed : int
            Random seed to ensure reproducible randomness.

        kwargs : dict
            Additional parameters for custom generator logic.

        Returns
        -------
        A simulated dataset, typically of the same structure as `base_data`.
        """
        pass

    def get_base_data(self):
        """
        Returns a copy of the base data. Subclasses can override this
        for preprocessing or transformation prior to modification.

        Returns
        -------
        A (shallow) copy of the base dataset.
        """
        return self.base_data.copy()
