from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """
    Base class for all estimators used in the POwer and COverage (poco) framework.
    """

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

    def is_covered(self, true_value: float = 0.0) -> bool:
        """
        Checks whether the confidence interval includes the true value (i.e., coverage).
        """
        return self.ci_lower <= true_value <= self.ci_upper
