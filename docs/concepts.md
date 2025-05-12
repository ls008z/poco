> **Disclaimer:** This documentation was generated with the assistance of AI and has not been thoroughly proofread yet. Content may contain inaccuracies or inconsistencies.

# Key Concepts

This document explains the core concepts behind Poco's simulation engine for power and cover curve analysis.

## Power Curves

A power curve represents the statistical power of a test or estimator across different parameter values. In the context of Poco:

- **Power**: The probability that a test correctly rejects a false null hypothesis
- **Parameters**: Variables that can be adjusted in the simulation (e.g., sample size, effect size)
- **Visualization**: Poco will generate plots showing how power changes as parameters vary

## Cover Curves

A cover curve shows the coverage probability of confidence intervals produced by different estimators:

- **Coverage**: The proportion of times a confidence interval contains the true parameter value
- **Confidence Level**: The nominal probability that the interval contains the true value (e.g., 95%)
- **Comparison**: Cover curves allow comparison of different estimation methods

## Estimators

Estimators are statistical methods for inferring parameters from data:

- **Point Estimators**: Provide a single "best guess" for a parameter
- **Interval Estimators**: Provide a range of plausible values (confidence intervals)
- **Properties**: Estimators can be evaluated on bias, variance, consistency, and efficiency

## Simulation Engine

Poco's simulation engine will:

1. Generate synthetic data according to user specifications
2. Apply different estimators to the data
3. Evaluate performance metrics (power, coverage, etc.)
4. Visualize results through power and cover curves
5. Allow comparison across different estimators and conditions