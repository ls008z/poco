from simulation.runner import SimulationRunner
from simulation.plotter import SimulationPlotter
from estimators.sample_mean import SampleMeanEstimator
from generators.bootstrap_constant_shifter import BootstrapConstantShifter
from generators.normal_mean_shifter import NormalMeanShifter
import pandas as pd
import numpy as np

# Simulated dataset
df = pd.DataFrame({"y": np.random.normal(loc=0, scale=1, size=100)})

# Initialize estimator + generator
estimator = SampleMeanEstimator(column="y", confidence=0.95)
# generator = BootstrapConstantShifter(base_data=df, outcome_col="y")
generator = NormalMeanShifter(n_samples=100, baseline_mean=0, std=1)

# Run simulations
runner = SimulationRunner(
    estimator=estimator,
    generator=generator,
    effect_sizes=np.linspace(0, 0.5, 20),
    n_simulations=500,
    null_value=0.0,
    seed=24,
    n_jobs=-1,
)
runner.run()

plotter = SimulationPlotter(runner)

# Power Curve
fig1 = plotter.plot_power_curve()
fig1.show()

# Coverage Curve
fig2 = plotter.plot_coverage_curve()
fig2.show()

# Estimate Distribution
fig3 = plotter.plot_estimate_distribution(effect_size=0.5)
fig3.show()

# Standard Error Histogram + MCSE overlay
fig4 = plotter.plot_standard_error_distribution(effect_size=0.5)
fig4.show()

# Bias Curve: mean estimate - true effect
fig5 = plotter.plot_bias_curve()
fig5.show()

# Standard Error Bias Curve: mean(estimator SE) - std(estimates)
fig6 = plotter.plot_se_bias_curve()
fig6.show()
