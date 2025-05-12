import numpy as np
import pandas as pd
from poco.generators.empirical.empirical_base import BaseEmpiricalGenerator


class PanelBlockTreatment(BaseEmpiricalGenerator):
    """
    Empirical generator for panel data. Randomly assigns a portion of entities to treatment,
    starting at a fixed time, and applies an effect to the outcome column.

    Supports both additive and multiplicative treatment effects.
    """

    def __init__(
        self,
        base_data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        treatment_col: str,
        outcome_col: str,
        treatment_start_time,
        treated_portion: float,
        mode: str = "additive",
    ):
        """
        Parameters
        ----------
        base_data : pd.DataFrame
            The panel dataset.

        entity_col : str
            Identifier for units (e.g., user ID, group ID).

        time_col : str
            Time identifier (e.g., date column).

        treatment_col : str
            Binary column to hold treatment status (1 = treated, 0 = control).

        outcome_col : str
            Column where the effect size will be applied post-treatment.

        treatment_start_time : str or pd.Timestamp
            Time from which treatment begins (inclusive). Will be converted to pd.Timestamp.

        treated_portion : float
            Fraction of entities to treat (between 0 and 1).

        mode : str
            Either "additive" or "multiplicative". Determines how effect_size is applied.
        """
        super().__init__(base_data)
        if not 0 <= treated_portion <= 1:
            raise ValueError("treated_portion must be between 0 and 1")
        if mode not in ("additive", "multiplicative"):
            raise ValueError("mode must be 'additive' or 'multiplicative'")

        self.entity_col = entity_col
        self.time_col = time_col
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.treatment_start_time = pd.to_datetime(treatment_start_time)
        self.treated_portion = treated_portion
        self.mode = mode

    def generate(self, effect_size: float, seed: int, **kwargs) -> pd.DataFrame:
        """
        Apply a treatment effect to a portion of entities starting at a fixed time.

        Parameters
        ----------
        effect_size : float
            The treatment effect. Interpretation depends on mode:
            - additive: added to outcome
            - multiplicative: multiplies outcome by (1 + effect_size)

        seed : int
            Random seed to ensure reproducibility.

        Returns
        -------
        pd.DataFrame
            Modified dataset with treatment assignments and updated outcomes.
        """
        rng = np.random.default_rng(seed)
        data = self.base_data.copy()

        # Ensure datetime consistency for time column
        data[self.time_col] = pd.to_datetime(data[self.time_col])

        # Sample treated entities
        all_entities = data[self.entity_col].unique()
        n_treated = int(len(all_entities) * self.treated_portion)
        treated_entities = rng.choice(all_entities, size=n_treated, replace=False)

        # Assign treatment indicator
        data[self.treatment_col] = 0
        treated_mask = data[self.entity_col].isin(treated_entities) & (
            data[self.time_col] >= self.treatment_start_time
        )
        data.loc[treated_mask, self.treatment_col] = 1

        # Apply treatment effect to outcome
        if self.mode == "additive":
            data.loc[treated_mask, self.outcome_col] += effect_size
        elif self.mode == "multiplicative":
            data.loc[treated_mask, self.outcome_col] *= 1 + effect_size

        return data.reset_index(drop=True)
