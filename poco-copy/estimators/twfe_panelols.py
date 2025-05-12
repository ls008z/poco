from linearmodels.panel import PanelOLS
from poco.estimators.base import BaseEstimator


class PanelOLSEstimator(BaseEstimator):
    def __init__(
        self,
        outcome="y",
        treatment="d",
        entity="entity",
        time="time",
        confidence=0.95,
        to_percentage=False,
    ):
        super().__init__(confidence=confidence)
        self.outcome = outcome
        self.treatment = treatment
        self.entity = entity
        self.time = time
        self.to_percentage = to_percentage
        self._results = None

    def fit(self, data, **kwargs):
        df = data.set_index([self.entity, self.time])
        y = df[self.outcome]
        X = df[[self.treatment]]

        model = PanelOLS(y, X, entity_effects=True, time_effects=True)
        self._results = model.fit(
            cov_type="unadjusted",
            # cov_type="clustered",
            # cluster_entity=True,
        )

        if self.to_percentage:
            self.get_percentage_lift(data)

    @property
    def confidence(self):
        return self._confidence

    @property
    def point_estimate(self):
        if self.to_percentage:
            return self._pct_point_estimate
        else:
            return self._results.params[self.treatment]

    @property
    def ci_lower(self):
        if self.to_percentage:
            return self._pct_ci_lower
        else:
            ci = self._results.conf_int(level=self.confidence)
            return ci.loc[self.treatment, "lower"]

    @property
    def ci_upper(self):
        if self.to_percentage:
            return self._pct_ci_upper
        else:
            ci = self._results.conf_int(level=self.confidence)
            return ci.loc[self.treatment, "upper"]

    @property
    def standard_error(self):
        if self.to_percentage:
            return self._pct_standard_error
        else:
            return self._results.std_errors[self.treatment]

    def get_percentage_lift(self, data):
        counterfactual_mean = data.loc[data[self.treatment] == 1, self.outcome].mean()

        self._pct_point_estimate = (
            self._results.params[self.treatment] / counterfactual_mean
        )
        self._pct_standard_error = (
            self._results.std_errors[self.treatment] / counterfactual_mean
        )
        ci = self._results.conf_int(level=self.confidence)
        self._pct_ci_lower = ci.loc[self.treatment, "lower"] / counterfactual_mean
        self._pct_ci_upper = ci.loc[self.treatment, "upper"] / counterfactual_mean


if __name__ == "__main__":
    from dev.eda.try_estimators.fake_panel_data import generate_mock_twfe_data

    df = generate_mock_twfe_data()
    estimator = PanelOLSEstimator(confidence=0.90)
    estimator.fit(df)
    print("Point Estimate:", estimator.point_estimate)
    print("CI:", estimator.ci_lower, estimator.ci_upper)
    print("SE:", estimator.standard_error)
