import pandas as pd
import plotly.express as px
from simulation.runner import SimulationRunner


class SimulationPlotter:
    """
    Visualization class for simulation results from a SimulationRunner.
    """

    def __init__(self, runner: SimulationRunner):
        """
        Parameters
        ----------
        runner : SimulationRunner
            A runner that has already completed simulations and has `.results` set.
        """
        if runner.results is None:
            raise ValueError("Runner has no results. Run `.run()` first.")
        self.runner = runner
        self.df = runner.results

    def plot_power_curve(self):
        """
        Plot the power curve — proportion of significant results by effect size.
        """
        alpha = 1 - self.runner.estimator.confidence

        power_data = (
            self.df.groupby("effect_size")["is_significant"]
            .mean()
            .reset_index(name="power")
        )
        fig = px.line(
            power_data,
            x="effect_size",
            y="power",
            title="Power Curve",
            labels={"effect_size": "Effect Size", "power": "Power"},
            markers=True,
        )
        fig.add_hline(
            y=alpha,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Significance Level (α = {alpha:.2f})",
            annotation_position="bottom right",
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), template="plotly_white")
        return fig

    def plot_coverage_curve(self):
        """
        Plot the coverage curve — proportion of simulations where CI covered the true effect.
        """
        conf = self.runner.estimator.confidence

        coverage_data = (
            self.df.groupby("effect_size")["is_covered"]
            .mean()
            .reset_index(name="coverage")
        )
        fig = px.line(
            coverage_data,
            x="effect_size",
            y="coverage",
            title="Coverage Curve",
            labels={"effect_size": "Effect Size", "coverage": "Coverage"},
            markers=True,
        )
        fig.add_hline(
            y=conf,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Nominal Coverage ({int(conf * 100)}%)",
            annotation_position="bottom right",
        )
        fig.update_layout(yaxis=dict(range=[0, 1]), template="plotly_white")
        return fig

    def plot_estimate_distribution(self, effect_size: float):
        """
        Plot histogram of point estimates at a specific effect size.

        Parameters
        ----------
        effect_size : float
            The effect size to filter simulations on.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive histogram of point estimates.
        """
        df_filtered = self.df[self.df["effect_size"] == effect_size]

        fig = px.histogram(
            df_filtered,
            x="estimate",
            nbins=30,
            title=f"Estimate Distribution (Effect Size = {effect_size})",
            labels={"estimate": "Point Estimate"},
        )

        # Add vertical line for true effect
        fig.add_vline(
            x=effect_size,
            line_dash="dash",
            line_color="red",
            annotation_text="True Effect",
        )

        fig.update_layout(template="plotly_white")
        return fig

    def plot_standard_error_distribution(self, effect_size: float):
        """
        Plot histogram of standard errors reported by the estimator, and overlay
        the Monte Carlo standard error of the estimates.

        Parameters
        ----------
        effect_size : float
            The effect size to filter simulations on.

        Returns
        -------
        plotly.graph_objects.Figure
            Histogram of standard errors with a line showing the MCSE of estimates.
        """
        df_filtered = self.df[self.df["effect_size"] == effect_size]
        mcse = df_filtered["estimate"].std(ddof=1)

        fig = px.histogram(
            df_filtered,
            x="standard_error",
            nbins=30,
            title=f"Standard Error Distribution (Effect Size = {effect_size})",
            labels={"standard_error": "Standard Error (from estimator)"},
        )

        fig.add_vline(
            x=mcse,
            line_dash="dot",
            line_color="blue",
            annotation_text="MC Std. Error",
            annotation_position="top left",
        )

        fig.update_layout(template="plotly_white")
        return fig

    def plot_bias_curve(self):
        """
        Plot the bias curve: (mean estimate - true effect) at each effect size.
        """
        bias_data = (
            self.df.groupby("effect_size")["estimate"]
            .mean()
            .reset_index(name="mean_estimate")
        )
        bias_data["bias"] = bias_data["mean_estimate"] - bias_data["effect_size"]

        fig = px.line(
            bias_data,
            x="effect_size",
            y="bias",
            title="Bias Curve",
            labels={"effect_size": "Effect Size", "bias": "Mean Bias"},
            markers=True,
        )
        fig.add_hline(
            y=0, line_dash="dot", line_color="gray", annotation_text="No Bias"
        )
        fig.update_layout(template="plotly_white")
        return fig

    def plot_se_bias_curve(self):
        """
        Plot the standard error bias curve:
        (mean reported SE - empirical std of estimates) at each effect size.
        """

        def se_bias(group):
            se_mean = group["standard_error"].mean()
            mc_std = group["estimate"].std(ddof=1)
            return pd.Series(
                {
                    "mean_se": se_mean,
                    "mc_std": mc_std,
                    "se_bias": se_mean - mc_std,
                }
            )

        se_data = self.df.groupby("effect_size").apply(se_bias).reset_index()

        fig = px.line(
            se_data,
            x="effect_size",
            y="se_bias",
            title="Standard Error Bias Curve",
            labels={
                "effect_size": "Effect Size",
                "se_bias": "SE Bias (Mean SE - MC Std)",
            },
            markers=True,
        )
        fig.add_hline(
            y=0, line_dash="dot", line_color="gray", annotation_text="No SE Bias"
        )
        fig.update_layout(template="plotly_white")
        return fig


if __name__ == "__main__":
    from simulation.runner import SimulationRunner
    from estimators.sample_mean import SampleMeanEstimator
    from generators.bootstrap_constant_shifter import BootstrapConstantShifter
    import pandas as pd
    import numpy as np

    # Simulated dataset
    df = pd.DataFrame({"y": np.random.normal(loc=0, scale=1, size=100)})

    # Initialize estimator + generator
    estimator = SampleMeanEstimator(column="y", confidence=0.95)
    generator = BootstrapConstantShifter(base_data=df, outcome_col="y")

    # Run simulations
    runner = SimulationRunner(
        estimator=estimator,
        generator=generator,
        effect_sizes=np.linspace(0, 0.5, 20),
        n_simulations=5000,
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
