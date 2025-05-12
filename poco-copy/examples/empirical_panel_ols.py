from poco.simulation.runner import SimulationRunner
from poco.simulation.plotter import SimulationPlotter
from poco.estimators.twfe_panelols import PanelOLSEstimator
from poco.generators.empirical.panel_data_treatment_randomization import (
    PanelBlockTreatment,
)
import pandas as pd
import numpy as np
import pickle

data_path = "local/data/low_asp_eda/tokyo_balanced_january.csv"
data = pd.read_csv(data_path)

null_value = 0
seed = 24
effect_sizes = np.linspace(-0.05, 0.05, 21)
n_simulations = 100
n_jobs = -1
h_1 = "!="


# # Initialize estimator + generator
# estimator = PanelOLSEstimator(
#     outcome="retail_ops",
#     treatment="treatment",
#     entity="zip7",
#     time="ship_day",
#     confidence=0.95,
#     to_percentage=True,
# )


# generator = PanelBlockTreatment(
#     data,
#     entity_col="zip7",
#     time_col="ship_day",
#     treatment_col="treatment",
#     outcome_col="retail_ops",
#     treatment_start_time="2025-01-15",
#     treated_portion=0.5,
#     mode="multiplicative",
# )

# runner = SimulationRunner(
#     estimator=estimator,
#     generator=generator,
#     effect_sizes=effect_sizes,
#     n_simulations=n_simulations,
#     null_value=null_value,
#     seed=seed,
#     n_jobs=n_jobs,
#     h_1=h_1,
# )

# results = runner.run()


# with open("local/junk/runner_tokyo_not_clustered.pkl", "wb") as file:
#     pickle.dump(runner, file)


with open("local/junk/runner_tokyo.pkl", "rb") as file:
    runner = pickle.load(file)

with open("local/junk/runner_tokyo_not_clustered.pkl", "rb") as file:
    runner_2 = pickle.load(file)


plotter = SimulationPlotter({"clustered": runner, "not_clustered": runner_2})

# Power Curve
plotter.plot_power_curve().show()
# Coverage Curve
plotter.plot_coverage_curve().show()
# Estimate Distribution
plotter.plot_estimate_distribution(effect_size=0).show()
# Standard Error Histogram + MCSE overlay
plotter.plot_standard_error_distribution(effect_size=0.05).show()
# Bias Curve: mean estimate - true effect
plotter.plot_bias_curve().show()
# Standard Error Bias Curve: mean(estimator SE) - std(estimates)
plotter.plot_se_bias_curve().show()
# MC Distribution
plotter.plot_mc_distribution().show()
# Ranked Estimates
plotter.plot_ranked_estimates(
    runner_label="not_clustered", effect_size=effect_sizes[3], n_max=100
).show()
