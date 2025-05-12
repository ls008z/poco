from abc import ABC, abstractmethod


class BaseDataGenerator(ABC):
    """
    Abstract base class for data generators used in the POwer and COverage (poco) framework.

    Subclasses must implement the `generate()` method, which produces a dataset given an effect size
    and a seed for reproducible randomness.
    """

    @abstractmethod
    def generate(self, effect_size: float, seed: int, **kwargs):
        """
        Generate a dataset that reflects a specific effect size using the given random seed.

        Parameters
        ----------
        effect_size : float
            The parameter controlling the signal (e.g., treatment effect, mean difference).

        seed : int
            Random seed to ensure reproducible results.

        kwargs : dict
            Optional extra arguments for flexible data generation.

        Returns
        -------
        data : object
            A user-defined data structure, typically a pandas DataFrame.
        """
        pass
