from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """
    Base class for all estimators used in the POwer and COverage (poco) framework.
    """

    def __init__(self, confidence: float):
        """
        Initialize the BaseEstimator with a confidence level.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.95 for 95% confidence).
        """
        if not (0 < confidence < 1):
            raise ValueError("Confidence must be a value between 0 and 1.")
        self._confidence = confidence

    @property
    @abstractmethod
    def confidence(self):
        """
        Returns the confidence level of the estimator.
        """
        return self._confidence

    @abstractmethod
    def fit(self, data, **kwargs):
        """
        Fit the estimator on a dataset.

        Parameters
        ----------
        data : Any
            User-defined data structure (e.g., DataFrame, dict, etc.)
        kwargs : dict
            Additional keyword arguments
        """
        pass

    @property
    @abstractmethod
    def point_estimate(self):
        """
        Returns the point estimate (e.g., coefficient, treatment effect).
        """
        pass

    @property
    @abstractmethod
    def ci_lower(self):
        """
        Returns the lower bound of the confidence interval.
        """
        pass

    @property
    @abstractmethod
    def ci_upper(self):
        """
        Returns the upper bound of the confidence interval.
        """
        pass

    @property
    @abstractmethod
    def standard_error(self):
        """
        Returns the standard error of the estimate.

        This is used for diagnostic plots and CI width analysis.
        """
        pass
