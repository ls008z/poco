from poco.simulation.runner import SimulationRunner
from poco.simulation.plotter import SimulationPlotter
from poco.estimators.twfe_panelols import PanelOLSEstimator
from poco.generators.parametric.panel_data_block_treatment import PanelDataGenerator
import pandas as pd
import numpy as np

n_entities = 100
n_periods = 10
common_start_time = 5
treated_share = 0.5
serial_corr = 0.8

null_value = 0
seed = 24
effect_sizes = np.linspace(-1, 1, 41)
n_simulations = 100
n_jobs = -1
h_1 = "!="


# Initialize estimator + generator
estimator = PanelOLSEstimator(
    outcome="y", treatment="d", entity="entity", time="time", confidence=0.95
)

generator = PanelDataGenerator(
    n_entities=n_entities,
    n_periods=n_periods,
    common_start_time=common_start_time,
    treated_share=treated_share,
    serial_corr=serial_corr,
)

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
