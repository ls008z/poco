import pandas as pd
import plotly.express as px
from poco.simulation.runner import SimulationRunner

import pandas as pd
import plotly.express as px
from poco.simulation.runner import SimulationRunner
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go


class SimulationPlotter:
    """
    Visualization class for simulation results from one or more SimulationRunners.
    Can generate overlayed plots for multiple runners.
    """

    def __init__(self, runners, default_label="Default"):
        """
        Parameters
        ----------
        runners : dict or SimulationRunner
            A dictionary of SimulationRunner instances with labels as keys or a single SimulationRunner.
        default_label : str, optional
            Label to use if a single SimulationRunner is provided (default is "Default").
        """
        # Initialize runners
        if isinstance(runners, SimulationRunner):
            self.runners = {default_label: runners}
        elif isinstance(runners, dict):
            if not all(isinstance(r, SimulationRunner) for r in runners.values()):
                raise TypeError(
                    "All values in the runners dictionary must be SimulationRunner instances."
                )
            self.runners = runners
        else:
            raise TypeError(
                "'runners' must be a SimulationRunner or a dictionary of runners."
            )

        # Validate results
        for label, runner in self.runners.items():
            if runner.results is None:
                raise ValueError(
                    f"Runner '{label}' has no results. Ensure `.run()` is executed first."
                )

        # Consistency check: All runners must share the same h_1 and confidence
        first_runner = next(iter(self.runners.values()))
        self.h_1 = first_runner.h_1
        self.confidence = first_runner.confidence

        for label, runner in self.runners.items():
            if runner.h_1 != self.h_1:
                raise ValueError(
                    f"Runner '{label}' has a different h_1 value ({runner.h_1}) than the first runner ({self.h_1})."
                )
            if runner.confidence != self.confidence:
                raise ValueError(
                    f"Runner '{label}' has a different confidence level ({runner.confidence}) than the first runner ({self.confidence})."
                )

        # Initialize figures dictionary
        self.figs: dict[str, px.Figure] = {}

    def _register_fig(self, name: str, fig):
        self.figs[name] = fig
        return fig

    def plot_power_curve(self):
        alpha = 1 - self.confidence if self.h_1 == "!=" else (1 - self.confidence) / 2

        fig = go.Figure()

        for label, runner in self.runners.items():
            df = runner.results
            power_data = (
                df.groupby("effect_size")["is_significant"]
                .mean()
                .reset_index(name="power")
            )

            fig.add_trace(
                go.Scatter(
                    x=power_data["effect_size"],
                    y=power_data["power"],
                    mode="lines+markers",
                    name=label,
                )
            )

        fig.add_hline(
            y=alpha,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Significance Level (α = {alpha:.4f})",
            annotation_position="bottom right",
        )

        fig.update_layout(
            title="Power Curve",
            xaxis_title="Effect Size",
            yaxis_title="Power",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
        )

        return self._register_fig("power_curve", fig)

    def plot_coverage_curve(self):
        fig = go.Figure()

        for label, runner in self.runners.items():
            df = runner.results
            coverage_data = (
                df.groupby("effect_size")["is_covered"]
                .mean()
                .reset_index(name="coverage")
            )

            fig.add_trace(
                go.Scatter(
                    x=coverage_data["effect_size"],
                    y=coverage_data["coverage"],
                    mode="lines+markers",
                    name=label,
                )
            )

        fig.add_hline(
            y=self.confidence,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Nominal Coverage ({int(self.confidence * 100)}%)",
            annotation_position="bottom right",
        )

        fig.update_layout(
            title="Coverage Curve",
            xaxis_title="Effect Size",
            yaxis_title="Coverage",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
        )

        return self._register_fig("coverage_curve", fig)

    def plot_estimate_distribution(self, effect_size: float):
        fig = go.Figure()

        for label, runner in self.runners.items():
            df_filtered = runner.results[runner.results["effect_size"] == effect_size]

            fig.add_trace(
                go.Histogram(
                    x=df_filtered["estimate"], nbinsx=30, name=label, opacity=0.6
                )
            )

        fig.add_vline(
            x=effect_size,
            line_dash="dash",
            line_color="red",
            annotation_text="True Effect",
        )

        fig.update_layout(
            title=f"Estimate Distribution (Effect Size = {effect_size})",
            xaxis_title="Point Estimate",
            yaxis_title="Frequency",
            barmode="overlay",
            template="plotly_white",
        )

        return self._register_fig(f"estimate_distribution_{effect_size}", fig)

    def plot_standard_error_distribution(self, effect_size: float):
        fig = go.Figure()

        for label, runner in self.runners.items():
            df_filtered = runner.results[runner.results["effect_size"] == effect_size]
            mcse = df_filtered["estimate"].std(ddof=1)

            fig.add_trace(
                go.Histogram(
                    x=df_filtered["standard_error"], nbinsx=30, name=label, opacity=0.6
                )
            )
            fig.add_vline(
                x=mcse,
                line_dash="dot",
                line_color="blue",
                annotation_text=f"MC SE ({label})",
                annotation_position="top left",
            )

        fig.update_layout(
            title=f"Standard Error Distribution (Effect Size = {effect_size})",
            xaxis_title="Standard Error (from estimator)",
            yaxis_title="Frequency",
            barmode="overlay",
            template="plotly_white",
        )

        return self._register_fig(f"standard_error_distribution_{effect_size}", fig)

    def plot_bias_curve(self):
        fig = go.Figure()

        for label, runner in self.runners.items():
            df = runner.results
            bias_data = (
                df.groupby("effect_size")["estimate"]
                .mean()
                .reset_index(name="mean_estimate")
            )
            bias_data["bias"] = bias_data["mean_estimate"] - bias_data["effect_size"]

            fig.add_trace(
                go.Scatter(
                    x=bias_data["effect_size"],
                    y=bias_data["bias"],
                    mode="lines+markers",
                    name=label,
                )
            )

        fig.add_hline(
            y=0, line_dash="dot", line_color="gray", annotation_text="No Bias"
        )

        fig.update_layout(
            title="Bias Curve",
            xaxis_title="Effect Size",
            yaxis_title="Mean Bias",
            template="plotly_white",
        )

        return self._register_fig("bias_curve", fig)

    def plot_se_bias_curve(self):
        fig = go.Figure()

        for label, runner in self.runners.items():
            df = runner.results

            # Separate groupby for mean_se and mc_std
            mean_se_data = (
                df.groupby("effect_size", as_index=False)["standard_error"]
                .mean()
                .rename(columns={"standard_error": "mean_se"})
            )
            mc_std_data = (
                df.groupby("effect_size", as_index=False)["estimate"]
                .std(ddof=1)
                .rename(columns={"estimate": "mc_std"})
            )

            # Merge the two dataframes
            se_data = pd.merge(mean_se_data, mc_std_data, on="effect_size")
            se_data["se_bias"] = se_data["mean_se"] - se_data["mc_std"]

            fig.add_trace(
                go.Scatter(
                    x=se_data["effect_size"],
                    y=se_data["se_bias"],
                    mode="lines+markers",
                    name=label,
                )
            )

        fig.add_hline(
            y=0, line_dash="dot", line_color="gray", annotation_text="No SE Bias"
        )

        fig.update_layout(
            title="Standard Error Bias Curve",
            xaxis_title="Effect Size",
            yaxis_title="SE Bias (Mean SE - MC Std)",
            template="plotly_white",
        )

        return self._register_fig("se_bias_curve", fig)

    def plot_mc_distribution(self, runner_label=None, q_levels=None):
        if runner_label is None:
            runner_label = next(iter(self.runners))

        if runner_label not in self.runners:
            raise ValueError(
                f"Runner '{runner_label}' not found in the provided runners."
            )

        if q_levels is None:
            q_levels = [0.05, 0.25, 0.75, 0.95]

        if len(q_levels) != 4:
            raise ValueError("q_levels must be a list of four values: [q1, q2, q3, q4]")

        q1, q2, q3, q4 = q_levels

        runner = self.runners[runner_label]
        df = runner.results

        summary = (
            df.groupby("effect_size")
            .agg(
                mean_estimate=("estimate", "mean"),
                q_low_outer=("estimate", lambda x: x.quantile(q1)),
                q_low_inner=("estimate", lambda x: x.quantile(q2)),
                q_high_inner=("estimate", lambda x: x.quantile(q3)),
                q_high_outer=("estimate", lambda x: x.quantile(q4)),
            )
            .reset_index()
        )

        fig = go.Figure()

        # Outer quantile band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([summary["effect_size"], summary["effect_size"][::-1]]),
                y=pd.concat([summary["q_high_outer"], summary["q_low_outer"][::-1]]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.05)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=f"Band ({q1:.2f}–{q4:.2f})",
            )
        )

        # Inner quantile band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([summary["effect_size"], summary["effect_size"][::-1]]),
                y=pd.concat([summary["q_high_inner"], summary["q_low_inner"][::-1]]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=f"Band ({q2:.2f}–{q3:.2f})",
            )
        )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=summary["effect_size"],
                y=summary["mean_estimate"],
                mode="lines+markers",
                name="MC Mean",
                line=dict(color="blue"),
            )
        )

        # y = x line
        min_x = df["effect_size"].min()
        max_x = df["effect_size"].max()
        fig.add_trace(
            go.Scatter(
                x=[min_x, max_x],
                y=[min_x, max_x],
                mode="lines",
                line=dict(color="red", dash="dot"),
                name="Perfect Estimation (y = x)",
            )
        )

        fig.update_layout(
            template="plotly_white",
            title=f"Monte Carlo Distribution Summary - {runner_label}",
            xaxis_title="True Effect Size",
            yaxis_title="Estimated Value",
        )

        return self._register_fig(f"mc_distribution_{runner_label}", fig)

    def plot_ranked_estimates(
        self, runner_label=None, effect_size: float = 0.0, n_max: int = 100
    ):
        if runner_label is None:
            runner_label = next(iter(self.runners))

        if runner_label not in self.runners:
            raise ValueError(
                f"Runner '{runner_label}' not found in the provided runners."
            )

        runner = self.runners[runner_label]
        conf = runner.confidence
        alpha = 1 - conf

        df_filtered = runner.results[
            runner.results["effect_size"] == effect_size
        ].copy()
        df_filtered = df_filtered.sort_values("estimate").reset_index(drop=True)

        # === Power Calculation ===
        power = (
            df_filtered["is_significant"].mean()
            if "is_significant" in df_filtered
            else 0
        )

        # === MC band (quantile) and MC mean ===
        q_low = df_filtered["estimate"].quantile(alpha / 2)
        q_high = df_filtered["estimate"].quantile(1 - alpha / 2)
        mc_mean = df_filtered["estimate"].mean()
        null_value = runner.null_value

        # === Downsample if needed ===
        n_total = len(df_filtered)
        if n_total > n_max:
            step = (n_total - 1) / (n_max - 1)
            indices = [round(i * step) for i in range(n_max)]
            df_filtered = df_filtered.iloc[indices].reset_index(drop=True)

        df_filtered["sim_index"] = df_filtered.index

        fig = go.Figure()

        # === MC Band as shaded region ===
        fig.add_trace(
            go.Scatter(
                x=[0, len(df_filtered) - 1, len(df_filtered) - 1, 0],
                y=[q_high, q_high, q_low, q_low],
                fill="toself",
                fillcolor="rgba(0,0,0,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name=f"MC Band ({int(conf * 100)}%)",
            )
        )

        # === Estimate points with CI ===
        fig.add_trace(
            go.Scatter(
                x=df_filtered["sim_index"],
                y=df_filtered["estimate"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=df_filtered["ci_upper"] - df_filtered["estimate"],
                    arrayminus=df_filtered["estimate"] - df_filtered["ci_lower"],
                    thickness=1.2,
                    width=4,
                ),
                mode="markers",
                marker=dict(size=6, color="black", opacity=0.7),
                name="Estimates",
            )
        )

        # === Reference Lines ===
        fig.add_trace(
            go.Scatter(
                x=[0, len(df_filtered) - 1],
                y=[effect_size, effect_size],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="True Effect",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, len(df_filtered) - 1],
                y=[mc_mean, mc_mean],
                mode="lines",
                line=dict(color="blue", dash="dot"),
                name="MC Mean",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, len(df_filtered) - 1],
                y=[null_value, null_value],
                mode="lines",
                line=dict(color="green", dash="dot"),
                name="Null Effect",
            )
        )

        # === Title ===
        title = (
            f"Ranked Estimates - {runner_label} (Effect Size = {effect_size})"
            f"<br>Power: {power:.3f} | Showing {len(df_filtered)} of {n_total} estimates"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Simulation (Ranked by Estimate)",
            yaxis_title="Estimate",
            template="plotly_white",
        )

        return self._register_fig(
            f"ranked_estimates_{runner_label}_{effect_size}_{n_max}", fig
        )


if __name__ == "__main__":
    from poco.simulation.runner import SimulationRunner
    from poco.estimators.sample_mean import SampleMeanEstimator
    from poco.generators.empirical.bootstrap_constant_shifter import (
        BootstrapConstantShifter,
    )
    from poco.generators.parametric.normal_mean_shifter import NormalMeanShifter
    import pandas as pd
    import numpy as np

    estimator = SampleMeanEstimator(column="y", confidence=0.95)

    # Simulated dataset
    # df = pd.DataFrame({"y": np.random.normal(loc=0, scale=1, size=200)})
    # generator = BootstrapConstantShifter(base_data=df, outcome_col="y")
    generator = NormalMeanShifter(n_samples=500)

    # Run simulations
    runner = SimulationRunner(
        estimator=estimator,
        generator=generator,
        effect_sizes=np.linspace(0, 0.5, 20),
        n_simulations=500,
        null_value=0.0,
        seed=10,
        n_jobs=-1,
    )
    runner.run()

    plotter = SimulationPlotter(runner)

    # test
    fig1 = plotter.plot_ranked_estimates()
    fig1.show()

    generator_2 = NormalMeanShifter(n_samples=200)

    runner_2 = SimulationRunner(
        estimator=estimator,
        generator=generator_2,
        effect_sizes=np.linspace(0, 1, 20),
        n_simulations=500,
        null_value=0.0,
        seed=24,
        n_jobs=-1,
    )
    runner_2.run()

    plotter_2 = SimulationPlotter({"big": runner, "small": runner_2})
    fig2 = plotter_2.plot_ranked_estimates("small")
    fig2.show()

    # plotter_2.plot_power_curve()
