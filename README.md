# Poco

A Python package for power and cover curve simulation with customizable estimators.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](https://github.com/ls008z/poco)

## Vision

Poco aims to be a comprehensive simulation engine for analyzing and visualizing:
- **Power curves**: Statistical power of tests across different parameters
- **Cover curves**: Coverage probability of confidence intervals
- **Estimator performance**: Comparison of different statistical estimators

*Note: Poco is currently in early development (v0.1.0) with minimal functionality. The full simulation engine is under development.*

## Features

Current:
- Basic package structure
- Command-line interface

Planned:
- Synthetic data generation
- Customizable statistical estimators
- Power curve visualization
- Cover curve analysis
- Estimator comparison tools

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/ls008z/poco.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/ls008z/poco.git
cd poco
pip install -e .
```

## Usage

### Current Usage

```python
from poco import main

main()  # Prints "Hello from poco!"
```

Or from the command line:

```bash
poco  # Prints "Hello from poco!"
```

### Future Usage (Planned)

```python
from poco import Simulator
from poco.estimators import OLS, RobustRegression
from poco.data import linear_model_generator

# Create simulator
sim = Simulator(random_seed=42)

# Define estimators to compare
estimators = [OLS(), RobustRegression()]

# Run simulations
results = sim.run(
    estimators=estimators,
    data_generator=linear_model_generator(outliers=True),
    n_simulations=1000
)

# Plot power curve
results.power_curve().plot(parameter='sample_size')

# Plot cover curve
results.cover_curve().plot(parameter='effect_size', confidence_level=0.95)
```

## Dependencies

- Python 3.8 or higher
- pandas 2.0.3 or higher

Future versions will likely require:
- numpy
- scipy
- matplotlib

## Documentation

Comprehensive documentation is available in the `docs/` directory. To build and view the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build and serve the documentation
cd docs
mkdocs serve
```

Then open your browser to http://127.0.0.1:8000/

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/ls008z/poco.git
cd poco

# Install development dependencies
pip install -e ".[dev]"
```

See the [Development Roadmap](docs/roadmap.md) for planned features and milestones.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.