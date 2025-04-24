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

from estimators.base import BaseEstimator
from generators.base import BaseDataGenerator


def run_simulation_task(args):
    return _simulate_task(*args)


def _simulate_task(
    estimator: BaseEstimator,
    generator: BaseDataGenerator,
    effect_size: float,
    seed: int,
    null_value: float,
) -> Dict[str, Any]:
    """
    A single simulation task. Generates data, fits estimator, and returns results.
    """
    import multiprocessing

    rng = np.random.default_rng(seed)
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

    return {
        "worker_id": worker_id,
        "effect_size": effect_size,
        "estimate": est,
        "standard_error": se,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "is_covered": ci_low <= effect_size <= ci_high,
        "is_significant": not (ci_low <= null_value <= ci_high),
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
        self.estimator = estimator
        self.generator = generator
        self.effect_sizes = effect_sizes
        self.n_simulations = n_simulations
        self.null_value = null_value
        self.seed = seed
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.results: pd.DataFrame | None = None

    def run(self) -> pd.DataFrame:
        """
        Run simulations across all effect sizes and replications.

        Returns
        -------
        pd.DataFrame
            Tidy dataframe of all simulation results with columns:
            - worker_id
            - effect_size
            - estimate
            - ci_lower
            - ci_upper
            - is_covered
            - is_significant
        """
        total_tasks = len(self.effect_sizes) * self.n_simulations

        print(f"\n🚀 Starting simulation...")
        print(f"• Total effect sizes: {len(self.effect_sizes)}")
        print(f"• Simulations per effect size: {self.n_simulations}")
        print(f"• Total simulations: {total_tasks}")
        print(f"• Parallel workers: {self.n_jobs}")
        print(f"• Random seed: {self.seed}\n")

        # Central RNG to generate unique seeds for each task
        master_rng = np.random.default_rng(self.seed)
        task_seeds = master_rng.integers(0, 2**32 - 1, size=total_tasks)

        tasks: List[tuple] = []
        idx = 0
        for effect_size in self.effect_sizes:
            for _ in range(self.n_simulations):
                seed = task_seeds[idx]
                tasks.append(
                    (self.estimator, self.generator, effect_size, seed, self.null_value)
                )
                idx += 1

        # Run tasks in parallel with progress bar
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results: List[Dict[str, Any]] = list(
                tqdm(
                    executor.map(run_simulation_task, tasks),
                    total=total_tasks,
                )
            )

        print("\n✅ Simulation complete.\n")
        self.results = pd.DataFrame(results)
        return self.results


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from estimators.sample_mean import SampleMeanEstimator
    from generators.bootstrap_constant_shifter import BootstrapConstantShifter

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
