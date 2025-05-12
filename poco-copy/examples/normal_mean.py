from poco.simulation.runner import SimulationRunner
from poco.simulation.plotter import SimulationPlotter
from poco.estimators.sample_mean import SampleMeanEstimator
from poco.generators.empirical.bootstrap_constant_shifter import (
    BootstrapConstantShifter,
)
from poco.generators.parametric.normal_mean_shifter import NormalMeanShifter
import pandas as pd
import numpy as np

n_samples = 200
null_value = 0
seed = 24
effect_sizes = np.linspace(-1, 1, 41)
n_simulations = 1000
n_jobs = -1
h_1 = "!="


# Initialize estimator + generator
estimator = SampleMeanEstimator(column="y", confidence=0.90)


# generator = NormalMeanShifter(n_samples=n_samples, baseline_mean=0, std=1)

# # Simulated dataset
np.random.seed(seed)
df = pd.DataFrame({"y": np.random.normal(loc=0, scale=1, size=n_samples)})
generator = BootstrapConstantShifter(base_data=df, outcome_col="y")


# Run simulations with Parametric Generator
runner = SimulationRunner(
    estimator=estimator,
    generator=generator,
    effect_sizes=effect_sizes,
    n_simulations=n_simulations,
    null_value=null_value,
    seed=seed,
    n_jobs=n_jobs,
    h_1=h_1,
)
results = runner.run()


plotter = SimulationPlotter(runner)

# Power Curve
plotter.plot_power_curve().show()
# Coverage Curve
plotter.plot_coverage_curve().show()
# Estimate Distribution
plotter.plot_estimate_distribution(effect_size=0).show()
# Standard Error Histogram + MCSE overlay
plotter.plot_standard_error_distribution(effect_size=0).show()
# Bias Curve: mean estimate - true effect
plotter.plot_bias_curve().show()
# Standard Error Bias Curve: mean(estimator SE) - std(estimates)
plotter.plot_se_bias_curve().show()
# MC Distribution
plotter.plot_mc_distribution().show()
# Ranked Estimates
plotter.plot_ranked_estimates(effect_size=0, n_max=100).show()
