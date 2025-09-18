# Inference Directory - Urban Information Project

This directory contains the enhanced vinference implementation with uncertainty-aware Bayesian p estimation and testing scripts for real Albuquerque data.

## Files Overview

### Core Implementation
- **`vinference.py`** - Main implementation with frequentist and Bayesian p estimation
- **`ssm_model.py`** - State Space Model for l_t inference

### Testing Scripts
- **`test_vinference.py`** - Test with dummy data (synthetic)
- **`test_albuquerque_vinference.py`** - Test with real Albuquerque CBSA ACS data
- **`demo_uncertainty.py`** - Demonstration of uncertainty handling with synthetic data

### Data Processing
- **`examine_albuquerque_data.py`** - Extract growth rates and uncertainties from Albuquerque data
- **`cbsa_acs_data.pkl`** - Raw Albuquerque CBSA ACS data (39MB)
- **`albuquerque_extracted_data.pkl`** - Processed growth rates and uncertainties

### Data Files
- **`dummy_data_*.pkl`** - Synthetic data for testing
- **`*_results.pkl`** - Results from various test runs

## Quick Start

### 1. Test with Dummy Data
```bash
cd inference
python test_vinference.py
```
This tests the vinference functions with synthetic data and shows the difference between frequentist and Bayesian approaches.

### 2. Test with Real Albuquerque Data
```bash
python test_albuquerque_vinference.py
```
This tests the vinference functions with real Albuquerque income data, including uncertainty-aware estimation.

### 3. Run Uncertainty Demo
```bash
python demo_uncertainty.py
```
This demonstrates how uncertainty-aware estimation handles borderline cases with synthetic data.

## Key Features

### Uncertainty-Aware Bayesian p Estimation
The enhanced `bayesian_p_estimation` function now handles:

- **Soft Classification**: Instead of hard binary wins/losses, calculates P(y_true > 0 | y_observed, y_std)
- **Fractional Wins**: Agents contribute fractional win probabilities based on measurement uncertainty
- **Multiple Uncertainty Methods**: Empirical, rolling, and constant uncertainty estimation
- **Backward Compatibility**: Falls back to hard classification if no uncertainties provided

### Real Data Testing
The Albuquerque testing provides:

- **Real Economic Patterns**: Actual urban income dynamics from ACS data
- **Population Weighting**: Real demographic patterns across block groups
- **Measurement Uncertainties**: Estimated from cross-sectional variance and temporal patterns
- **Method Comparison**: Side-by-side comparison of all estimation approaches

## Workflow

### For New Data
1. **Extract Data**: Run `examine_albuquerque_data.py` to process raw data
2. **Test Methods**: Use `test_albuquerque_vinference.py` to compare approaches
3. **Analyze Results**: Examine saved results and visualizations

### For Method Development
1. **Synthetic Testing**: Use `demo_uncertainty.py` for controlled experiments
2. **Dummy Data**: Use `test_vinference.py` for baseline validation
3. **Real Data**: Use Albuquerque data for realistic testing

## Uncertainty Estimation Methods

### 1. Empirical Method
- Uses standard deviation across block groups at each timestep
- Captures cross-sectional variance in growth rates
- Most realistic for urban economic data

### 2. Rolling Method
- Uses rolling standard deviation within each block group
- Captures temporal volatility for individual areas
- Good for areas with varying economic stability

### 3. Constant Method
- Uses overall variance across all data
- Simple baseline approach
- Good for comparison and validation

## Output Files

### Results
- **`vinference_test_results.pkl`** - Results from dummy data testing
- **`albuquerque_vinference_results.pkl`** - Results from Albuquerque data testing

### Visualizations
- **`uncertainty_demo_results.png`** - Synthetic data demonstration plots
- **`albuquerque_vinference_results.png`** - Albuquerque data analysis plots

## Key Insights

1. **Uncertainty Matters**: Growth rates near zero get fractional win probabilities
2. **Method Comparison**: Different approaches yield different results on real data
3. **Population Weighting**: More populous areas have greater influence on estimates
4. **Real Data Validation**: Albuquerque data shows realistic urban economic patterns

## Next Steps

1. **Implement Recommendations**: Address the issues identified in `upgrades.txt`
2. **Temporal Priors**: Add temporal structure to l_t estimation
3. **Hierarchical Models**: Model agent belief distributions more flexibly
4. **Cross-Validation**: Implement proper validation frameworks
5. **Scale Up**: Test with larger datasets and more complex environments

## Dependencies

- numpy
- pandas
- matplotlib
- scipy
- pymc (for SSM models)
- pickle (built-in)

## Notes

- The Albuquerque data contains real economic patterns that may differ from synthetic assumptions
- Population weights reflect actual demographic distributions
- Uncertainties are estimated from the data rather than provided a priori
- Results show how methods perform on realistic urban economic data 