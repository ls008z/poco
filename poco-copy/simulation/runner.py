"""
poco.simulation.runner

SimulationRunner class to evaluate estimators using repeated simulations
across a grid of effect sizes, with support for parallel processing and reproducibility.
"""

import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any
import copy
from tqdm import tqdm

from poco.estimators.base import BaseEstimator
from poco.generators.base import BaseDataGenerator
from scipy.stats import norm


def run_simulation_task(args):
    return _simulate_task(*args)


def _simulate_task(
    estimator: BaseEstimator,
    generator: BaseDataGenerator,
    effect_size: float,
    seed: int,
) -> Dict[str, Any]:
    """
    A single simulation task. Generates data, fits estimator, and returns results.
    """
    import multiprocessing

    worker_id: str = multiprocessing.current_process().name

    # Deep copy estimator and generator to isolate tasks
    estimator_copy = copy.deepcopy(estimator)
    generator_copy = copy.deepcopy(generator)

    # Generate and fit
    data = generator_copy.generate(effect_size=effect_size, seed=seed)
    estimator_copy.fit(data)

    est = estimator_copy.point_estimate
    se = estimator_copy.standard_error
    ci_low = estimator_copy.ci_lower
    ci_high = estimator_copy.ci_upper
    confidence = estimator.confidence

    return {
        "worker_id": worker_id,
        "effect_size": effect_size,
        "estimate": est,
        "standard_error": se,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "confidence": confidence,
    }


class SimulationRunner:
    def __init__(
        self,
        estimator: BaseEstimator,
        generator: BaseDataGenerator,
        effect_sizes: List[float],
        n_simulations: int = 100,
        null_value: float = 0.0,
        seed: int = 42,
        n_jobs: int = -1,
        h_1: str = "!=",
    ) -> None:
        """
        Initialize the simulation runner.

        Parameters
        ----------
        estimator : BaseEstimator
            A fully configured estimator instance (e.g., SampleMeanEstimator).
        generator : BaseDataGenerator
            A fully configured generator instance (e.g., BootstrapConstantShifter).
        effect_sizes : List[float]
            Values of the effect size to simulate.
        n_simulations : int
            Number of repetitions per effect size.
        null_value : float
            The value used to assess statistical significance (typically 0).
        seed : int
            Seed for centralized RNG for reproducibility.
        n_jobs : int
            Number of parallel processes to use (-1 = all available cores).
        """
        if h_1 not in ("!=", ">", "<"):
            raise ValueError("h_1 must be one of '!=', '>', or '<'")

        self.estimator = estimator
        self.generator = generator
        self.effect_sizes = effect_sizes
        self.n_simulations = n_simulations
        self.null_value = null_value
        self.seed = seed
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.h_1 = h_1
        self.results: pd.DataFrame | None = None
        self.confidence = self.estimator.confidence
        self.past_results = {}

    def run(self) -> pd.DataFrame:
        total_tasks = len(self.effect_sizes) * self.n_simulations

        print(f"\n🚀 Starting simulation...")
        print(f"• Total effect sizes: {len(self.effect_sizes)}")
        print(f"• Simulations per effect size: {self.n_simulations}")
        print(f"• Total simulations: {total_tasks}")
        print(f"• Parallel workers: {self.n_jobs}")

        master_rng = np.random.default_rng(self.seed)
        task_seeds = master_rng.integers(0, 2**32 - 1, size=total_tasks)

        tasks = [
            (self.estimator, self.generator, es, task_seeds[i])
            for i, es in enumerate(self.effect_sizes * self.n_simulations)
        ]

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(
                tqdm(executor.map(run_simulation_task, tasks), total=total_tasks)
            )

        df_results = pd.DataFrame(results)
        df_results["confidence"] = self.confidence
        df_results["h_1"] = self.h_1

        self.results = self._compute_coverage_and_significance(df_results)

        return self.results

    def _compute_coverage_and_significance(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_covered"] = (df["ci_lower"] <= df["effect_size"]) & (
            df["ci_upper"] >= df["effect_size"]
        )
        self._update_significance(df)
        return df

    def confidence_override(self, confidence: float) -> None:
        if self.results is not None:
            self.past_results[(self.confidence, self.h_1)] = self.results.copy()
            self.confidence = confidence

            z_value = norm.ppf(1 - (1 - confidence) / 2)
            self.results["ci_lower"] = (
                self.results["estimate"] - z_value * self.results["standard_error"]
            )
            self.results["ci_upper"] = (
                self.results["estimate"] + z_value * self.results["standard_error"]
            )
            self.results["confidence"] = confidence

            self.results = self._compute_coverage_and_significance(self.results)

    def hypothesis_override(self, h_1: str) -> None:
        if h_1 not in ("!=", ">", "<"):
            raise ValueError("h_1 must be one of '!=', '>', or '<'")

        if self.results is not None:
            self.past_results[(self.confidence, self.h_1)] = self.results.copy()
            self.h_1 = h_1
            self.results["h_1"] = h_1
            self._update_significance(self.results)

    def _update_significance(self, df: pd.DataFrame) -> None:
        if self.h_1 == "!=":
            df["is_significant"] = ~(
                (df["ci_lower"] <= self.null_value)
                & (self.null_value <= df["ci_upper"])
            )
        elif self.h_1 == ">":
            df["is_significant"] = df["ci_lower"] > self.null_value
        elif self.h_1 == "<":
            df["is_significant"] = df["ci_upper"] < self.null_value


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from poco.estimators.sample_mean import SampleMeanEstimator
    from poco.generators.empirical.bootstrap_constant_shifter import (
        BootstrapConstantShifter,
    )

    # Create some sample data
    df = pd.DataFrame({"y": np.random.normal(0, 1, 100)})

    # Initialize generator and estimator
    estimator = SampleMeanEstimator(column="y", confidence=0.95)
    generator = BootstrapConstantShifter(base_data=df, outcome_col="y")

    # Define simulation runner
    runner = SimulationRunner(
        estimator=estimator,
        generator=generator,
        effect_sizes=np.linspace(0, 1, 50),
        n_simulations=200,
        null_value=0.0,
        seed=42,
        n_jobs=-1,  # Safe for test runs
    )

    # Run and print result
    results = runner.run()
    print(results.head())
