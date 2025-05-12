import numpy as np
import scipy.stats as stats
import pandas as pd
from poco.estimators.base import BaseEstimator


class SampleMeanEstimator(BaseEstimator):
    def __init__(self, column: str, confidence: float = 0.95):
        """
        Parameters
        ----------
        column : str
            Name of the column to compute the mean of
        confidence : float
            Confidence level for CI (default 0.95)
        """
        super().__init__(confidence=confidence)
        self.column = column
        self._mean = None
        self._ci = (None, None)
        self._se = None

    def fit(self, data: pd.DataFrame, **kwargs):
        values = data[self.column].dropna().values
        n = len(values)
        mean = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(n)
        z = stats.norm.ppf(1 - (1 - self.confidence) / 2)
        ci_lower = mean - z * se
        ci_upper = mean + z * se

        self._mean = mean
        self._se = se
        self._ci = (ci_lower, ci_upper)

    @property
    def point_estimate(self):
        return self._mean

    @property
    def ci_lower(self):
        return self._ci[0]

    @property
    def ci_upper(self):
        return self._ci[1]

    @property
    def standard_error(self):
        return self._se

    @property
    def confidence(self):
        return self._confidence


if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({"outcome": np.random.normal(loc=1.0, scale=1.0, size=100)})

    est = SampleMeanEstimator(column="outcome")
    est.fit(df)

    print("Estimate:", est.point_estimate)
    print("SE:", est.standard_error)
    print("CI:", (est.ci_lower, est.ci_upper))
