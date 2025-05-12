> **Disclaimer:** This documentation was generated with the assistance of AI and has not been thoroughly proofread yet. Content may contain inaccuracies or inconsistencies.

# Installation

There are several ways to install Poco, depending on your needs.

## From GitHub

You can install the latest version directly from GitHub:

```bash
pip install git+https://github.com/ls008z/poco.git
```

## Local Installation

For development or to work with the source code:

```bash
# Clone the repository
git clone https://github.com/ls008z/poco.git
cd poco

# Install in development mode
pip install -e .
```

## Development Setup

If you plan to contribute to Poco, install the development dependencies:

```bash
pip install -e ".[dev]"
```

## Requirements

Poco requires:

- Python 3.8 or higher
- pandas 2.0.3 or higher

## Verification

To verify that Poco is installed correctly, run:

```bash
poco
```

You should see the message "Hello from poco!" printed to the console.