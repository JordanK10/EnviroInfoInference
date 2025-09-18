# EnviroInfoInference

A research project focused on **urban information inference** - analyzing urban income dynamics and socioeconomic mobility using Bayesian inference techniques.

## Project Structure

```
EnviroInfoInference/
├── src/                          # Core source code
│   ├── inference/               # Main inference engine
│   │   ├── vinference.py        # Variational inference implementation
│   │   ├── ssm_model.py         # State Space Model for l_t inference
│   │   └── utils.py             # Shared utilities and path management
│   ├── validation/              # Validation framework
│   │   ├── data_generation.py   # Synthetic data generation
│   │   ├── model_fitting.py     # VI model fitting
│   │   └── plotting.py          # Validation visualization
│   └── analysis/                # Real-world data analysis
│       ├── data_prep.py         # ACS data preprocessing
│       ├── wins_growth_correlation.py  # Growth vs wins correlation
│       └── temporal_autocorrelation.py # Temporal persistence analysis
├── data/                        # Data storage
│   ├── raw/                     # Raw ACS data
│   ├── processed/               # Processed datasets
│   └── validation/              # Validation datasets
├── notebooks/                   # Jupyter notebooks
│   ├── exploratory/             # Exploratory analysis
│   └── validation/              # Validation notebooks
├── figures/                     # Generated figures
│   ├── publication/             # Publication-ready figures
│   └── exploratory/             # Exploratory plots
├── tests/                       # Test suite
│   ├── integration/             # Integration tests
│   └── unit/                    # Unit tests
└── legacy/                      # Original codebase (preserved)
```

## Core Methodology

### Two-Stage Inference Process

1. **Environmental Predictability (p_t)**: Calculates fraction of agents with positive income growth
2. **Environmental Payouts (l_t)**: Uses p_t to infer l_t via State Space Model

### State Space Model (SSM)
- Models agent beliefs as Gaussian Random Walk in logit space
- Uses PyMC for Bayesian inference
- Handles uncertainty in growth rate measurements

## Quick Start

### 1. Validation Testing
```bash
cd src/validation
python data_generation.py  # Generate synthetic data
python model_fitting.py     # Fit VI models
python plotting.py          # Generate validation plots
```

### 2. Real Data Analysis
```bash
cd src/analysis
python data_prep.py                    # Preprocess ACS data
python wins_growth_correlation.py      # Analyze correlations
python temporal_autocorrelation.py     # Analyze temporal patterns
```

### 3. Inference Engine
```bash
cd src/inference
python vinference.py  # Run main inference pipeline
```

### 4. Testing
```bash
python tests/integration/test_dummy_data_workflow.py
```

## Dependencies

- numpy
- pandas
- matplotlib
- scipy
- pymc
- arviz
- joblib
- tqdm

## Key Features

- **Uncertainty-aware Bayesian estimation**
- **Population-weighted frequentist approach**
- **Comprehensive validation framework**
- **Real-world ACS data integration**
- **Publication-ready visualizations**
- **Standardized import management**

## Legacy Code

The original codebase is preserved in the `legacy/` directory for reference and comparison.

## Contributing

1. Follow the modular structure
2. Use `from inference.utils import setup_project_paths` for imports
3. Add tests for new functionality
4. Update documentation
5. Use consistent data interfaces

## Research Context

This project implements information inference techniques for analyzing urban income dynamics, with applications to socioeconomic mobility research.
