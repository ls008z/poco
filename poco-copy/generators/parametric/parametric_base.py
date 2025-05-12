from poco.generators.base import BaseDataGenerator


class BaseParametricGenerator(BaseDataGenerator):
    """
    A parametric generator defines a full data generating process (DGP),
    which is modified based on effect_size and controlled by an external seed.
    """

    def __init__(self):
        """
        Initialize a parametric generator.
        Subclasses can override or extend this if needed.
        """
        pass

    def generate(self, effect_size: float, seed: int, **kwargs):
        """
        Generate synthetic data from a specified DGP with the given effect size and seed.

        Parameters
        ----------
        effect_size : float
            A parameter controlling the simulated signal (e.g., treatment effect, mean difference).

        seed : int
            Random seed to ensure reproducible randomness.

        kwargs : dict
            Additional optional arguments for flexible model behavior.

        Returns
        -------
        A user-defined data structure, typically a pandas DataFrame.
        """
        raise NotImplementedError
