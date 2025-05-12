# API Reference

This document provides detailed information about the Poco API. As Poco evolves into a simulation engine for power and cover curve analysis, this API reference will be updated accordingly.

## Current API (v0.1.0)

### `main()`

```python
def main():
    """Print a greeting message."""
    print("Hello from poco!")
```

**Description:**  
Prints a friendly greeting message to the console. This is a placeholder function that will be replaced with the simulation engine functionality in future versions.

**Parameters:**  
None

**Returns:**  
None

**Example:**
```python
from poco import main
main()  # Prints "Hello from poco!"
```

## Planned API (Future Versions)

The following API is planned for future versions of Poco as it evolves into a simulation engine for power and cover curve analysis:

### `Simulator` Class

```python
class Simulator:
    """Simulation engine for power and cover curve analysis."""
    
    def __init__(self, random_seed=None):
        """Initialize the simulator.
        
        Args:
            random_seed (int, optional): Seed for random number generation
        """
        pass
        
    def run(self, estimators, data_generator, n_simulations=1000):
        """Run simulations for the given estimators and data generator.
        
        Args:
            estimators (list): List of estimator objects
            data_generator (callable): Function to generate synthetic data
            n_simulations (int): Number of simulations to run
            
        Returns:
            SimulationResults: Object containing simulation results
        """
        pass
```

### `PowerCurve` Class

```python
class PowerCurve:
    """Power curve visualization and analysis."""
    
    def __init__(self, simulation_results):
        """Initialize with simulation results.
        
        Args:
            simulation_results (SimulationResults): Results from simulation
        """
        pass
        
    def plot(self, parameter='sample_size', ax=None):
        """Plot the power curve.
        
        Args:
            parameter (str): Parameter to vary on x-axis
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        pass
```

### `CoverCurve` Class

```python
class CoverCurve:
    """Cover curve visualization and analysis."""
    
    def __init__(self, simulation_results):
        """Initialize with simulation results.
        
        Args:
            simulation_results (SimulationResults): Results from simulation
        """
        pass
        
    def plot(self, parameter='sample_size', confidence_level=0.95, ax=None):
        """Plot the cover curve.
        
        Args:
            parameter (str): Parameter to vary on x-axis
            confidence_level (float): Nominal confidence level
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plot
        """
        pass
```

## Module Structure

Current:
- `poco/__init__.py`: Package initialization, exports the `main` function
- `poco/core.py`: Contains the core functionality including the `main` function
- `poco/version.py`: Contains version information

Planned:
- `poco/simulator.py`: Simulation engine implementation
- `poco/estimators/`: Directory containing various estimator implementations
- `poco/visualization/`: Tools for plotting power and cover curves
- `poco/data/`: Data generation utilities