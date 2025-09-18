# Figure 3: L Inference from Real Data

This directory contains the script for Figure 3, which demonstrates the cross-sectional l inference protocol on real income data.

## Purpose

The script reads in income growth rate data and fitted p values from Figure 2, then runs the cross-sectional l inference protocol to retrieve a time series of l (outcomes multiplier) values.

## What it does

1. **Loads data** from Figure 2 (income growth rates and p values)
2. **Runs cross-sectional l inference** for each timestep using the `fit_l_cross_sectional()` function
3. **Applies temporal smoothing** to create a clean l time series
4. **Plots the results** showing raw estimates, uncertainties, and smoothed values
5. **Saves outputs** as PNG plot, pickle file, and CSV

## Key Functions Used

- `fit_l_cross_sectional()` - Cross-sectional Bayesian model for estimating l at a single timestep
- `estimate_l_time_series_bayesian()` - Full workflow with temporal smoothing

## Input Data Requirements

The script expects data with:
- Income growth rates (log growth rates)
- P values (fraction of positive growth rates) 
- Year/timestep information

## Output Files

- `figure3_l_time_series.png` - Visualization of l estimates over time
- `figure3_l_inference_results.pkl` - Full results in pickle format
- `figure3_l_inference_results.csv` - Results in CSV format for easy viewing

## Usage

```bash
cd figure3
python figure3.py
```

## Dependencies

- numpy, pandas, matplotlib, seaborn
- SSM model functions from validation directory
- Real income data from Figure 2

## Notes

- The script automatically searches for data in multiple possible locations
- It requires at least 5 data points per timestep for reliable l estimation
- Uses a 3-year rolling window for temporal smoothing by default
- Includes comprehensive error handling and progress reporting 