import numpy as np
import pandas as pd
from poco.generators.parametric.parametric_base import BaseParametricGenerator


class PanelDataGenerator(BaseParametricGenerator):
    """
    A panel data generator that directly implements the mock TWFE data generation logic
    for integration with the POCO framework.
    """

    def __init__(
        self,
        n_entities: int = 100,
        n_periods: int = 10,
        common_start_time: int = 5,
        treated_share: float = 0.5,
        serial_corr: float = 0.0,
    ):
        """
        Parameters
        ----------
        n_entities : int
            Number of entities in the panel.

        n_periods : int
            Number of time periods.

        common_start_time : int
            Time period at which treatment begins for treated units.

        treated_share : float
            Proportion of entities that receive treatment (between 0 and 1).

        serial_corr : float
            Autoregressive parameter for serial correlation in errors (AR(1) within entity).
        """
        self.n_entities = n_entities
        self.n_periods = n_periods
        self.common_start_time = common_start_time
        self.treated_share = treated_share
        self.serial_corr = serial_corr

    def generate(self, effect_size: float, seed: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic panel data with a specified treatment effect.

        Parameters
        ----------
        effect_size : float
            The treatment effect (beta) to be used in data generation.

        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ['entity', 'time', 'd', 'y'] representing
            the simulated panel data.
        """
        np.random.seed(seed)

        entities = np.arange(self.n_entities)
        times = np.arange(self.n_periods)
        panel = pd.MultiIndex.from_product([entities, times], names=["entity", "time"])
        df = pd.DataFrame(index=panel).reset_index()

        # Generate fixed effects
        entity_effect = np.random.normal(0, 1, self.n_entities)
        time_effect = np.random.normal(0, 1, self.n_periods)

        # Assign treatment
        n_treated = int(self.n_entities * self.treated_share)
        treated_entities = np.random.choice(entities, size=n_treated, replace=False)

        def assign_treatment(row):
            return int(
                row["entity"] in treated_entities
                and row["time"] >= self.common_start_time
            )

        df["d"] = df.apply(assign_treatment, axis=1)

        df["entity_effect"] = df["entity"].map(dict(zip(entities, entity_effect)))
        df["time_effect"] = df["time"].map(dict(zip(times, time_effect)))

        # Generate serially correlated errors (AR(1) process)
        epsilons = []
        for entity in entities:
            e = np.zeros(self.n_periods)
            e[0] = np.random.normal()
            for t in range(1, self.n_periods):
                e[t] = self.serial_corr * e[t - 1] + np.random.normal()
            epsilons.extend(e)
        df["epsilon"] = epsilons

        df["y"] = (
            df["entity_effect"]
            + df["time_effect"]
            + effect_size * df["d"]
            + df["epsilon"]
        )

        return df[["entity", "time", "d", "y"]]


if __name__ == "__main__":
    generator = PanelDataGenerator(
        n_entities=50,
        n_periods=8,
        common_start_time=3,
        treated_share=0.6,
        serial_corr=0.8,
    )
    df = generator.generate(effect_size=1.5, seed=123)
    print(df.head())
