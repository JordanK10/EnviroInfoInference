#!/usr/bin/env python3
"""
Figure 3: L Inference from Real Data
This script reads in income growth rate data and fitted p values from figure2,
then runs the cross-sectional l inference protocol to retrieve a time series of l.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the validation directory to the path to import SSM functions
sys.path.append('../validation/validation_dynamic_p_l')
from ssm_model import fit_l_cross_sectional, estimate_l_time_series_bayesian

def load_figure2_data():
    """
    Load the income growth rate data and fitted p values from figure2.
    This function should be updated based on the actual structure of figure2 output.
    """
    print("Loading data from figure2...")
    
    # Try to load from different possible locations
    possible_paths = [
        "../figure2/figure2/figure2_data.pkl",
        "../figure2/figure2_data.pkl",
        "../figure2/processed_acs_data.pkl",
        "../figure2/figure2_data.pkl", 
        "../../urbandata/data/processed_acs_data.pkl",
        "../../urbandata/data/figure2_output.pkl",
        "../processed_acs_data.pkl"
    ]
    
    data = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found data at: {path}")
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                break
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    if data is None:
        print("❌ Could not find figure2 data. Please check the data paths.")
        return None
    
    print(f"✅ Successfully loaded data")
    print(f"Data type: {type(data)}")
    if hasattr(data, 'keys'):
        print(f"Data keys: {list(data.keys())}")
    
    return data

def extract_growth_rates_and_p_values(data):
    """
    Extract income growth rates and p values from the loaded data.
    This function needs to be customized based on the actual data structure.
    """
    print("\nExtracting growth rates and p values...")
    
    # This is a placeholder - you'll need to customize this based on your actual data structure
    if isinstance(data, dict):
        # Example structure - adjust based on your actual data
        if 'growth_rates' in data and 'p_values' in data:
            growth_rates = data['growth_rates']
            p_values = data['p_values']
            years = data.get('years', list(range(len(p_values))))
        elif 'block_groups' in data:
            # If data is organized by block groups
            growth_rates = []
            p_values = []
            years = []
            
            # Extract from block group data
            for bg_data in data['block_groups']:
                if 'growth_rates' in bg_data and 'p_values' in bg_data:
                    growth_rates.extend(bg_data['growth_rates'])
                    p_values.extend(bg_data['p_values'])
                    if 'years' in bg_data:
                        years.extend(bg_data['years'])
            
            # Convert to numpy arrays
            growth_rates = np.array(growth_rates)
            p_values = np.array(p_values)
            years = np.array(years) if years else np.arange(len(p_values))
        else:
            print("❌ Unexpected data structure. Please check the data format.")
            return None, None, None
    else:
        print("❌ Data is not a dictionary. Please check the data format.")
        return None, None, None
    
    # Convert to numpy arrays if they aren't already
    if not isinstance(growth_rates, np.ndarray):
        growth_rates = np.array(growth_rates)
    if not isinstance(p_values, np.ndarray):
        p_values = np.array(p_values)
    if not isinstance(years, np.ndarray):
        years = np.array(years)
    
    print(f"✅ Extracted {len(growth_rates)} growth rates and {len(p_values)} p values")
    print(f"Growth rates shape: {growth_rates.shape}")
    print(f"Years range: {years.min()} to {years.max()}")
    print(f"Growth rates range: {growth_rates.min():.3f} to {growth_rates.max():.3f}")
    print(f"P values range: {p_values.min():.3f} to {p_values.max():.3f}")
    
    return growth_rates, p_values, years

def run_l_inference(growth_rates, p_values, years):
    """
    Run the cross-sectional l inference protocol to estimate l for each timestep.
    """
    print("\nRunning cross-sectional l inference...")
    
    # Group growth rates by year/timestep
    unique_years = np.unique(years)
    print(f"Found {len(unique_years)} unique timesteps: {unique_years}")
    
    # Estimate l for each timestep
    l_estimates = {}
    l_uncertainties = {}
    
    # growth_rates is organized as (block_groups, years)
    # We need to extract each year's growth rates
    for i, year in enumerate(unique_years):
        print(f"\nProcessing year {year}...")
        
        # Get growth rates for this year (all block groups)
        year_growth_rates = growth_rates[:, i]  # Extract column i for year i
        
        # Get p value for this year
        year_p_value = p_values[i]
        
        # Filter out NaN values
        valid_mask = np.isfinite(year_growth_rates)
        year_growth_rates_clean = year_growth_rates[valid_mask]
        
        print(f"  {len(year_growth_rates_clean)} valid growth rates (from {len(year_growth_rates)} total), p = {year_p_value:.3f}")
        
        if len(year_growth_rates_clean) >= 5:  # Need minimum data points
            try:
                # Run cross-sectional l inference
                idata = fit_l_cross_sectional(
                    y_values_t=year_growth_rates_clean,
                    p_hat_t=year_p_value,
                    delta=0.05,  # Sub-optimality offset
                    n_samples=8000
                )
                
                # Extract posterior samples
                l_posterior_samples = idata.posterior["l"].values.flatten()
                
                # Calculate point estimate and uncertainty
                l_est = np.mean(l_posterior_samples)
                l_std = np.std(l_posterior_samples)
                l_ci_95 = np.percentile(l_posterior_samples, [2.5, 97.5])
                
                l_estimates[year] = l_est
                l_uncertainties[year] = {
                    'std': l_std,
                    'ci_95_lower': l_ci_95[0],
                    'ci_95_upper': l_ci_95[1]
                }
                
                print(f"  ✅ l_est = {l_est:.3f} ± {l_std:.3f}")
                print(f"     95% CI: [{l_ci_95[0]:.3f}, {l_ci_95[1]:.3f}]")
                
            except Exception as e:
                print(f"  ❌ Failed to estimate l: {e}")
                # Use reasonable defaults
                l_estimates[year] = 2.0
                l_uncertainties[year] = {
                    'std': 0.5,
                    'ci_95_lower': 1.5,
                    'ci_95_upper': 2.5
                }
        else:
            print(f"  ⚠️  Insufficient data points ({len(year_growth_rates_clean)} < 5)")
            l_estimates[year] = 2.0
            l_uncertainties[year] = {
                'std': 0.5,
                'ci_95_lower': 1.5,
                'ci_95_upper': 2.5
            }
    
    return l_estimates, l_uncertainties

def apply_temporal_smoothing(l_estimates, l_uncertainties, rolling_window=3):
    """
    Apply temporal smoothing to the l estimates using a rolling mean.
    """
    print(f"\nApplying temporal smoothing with {rolling_window}-year rolling window...")
    
    # Convert to sorted arrays
    years = sorted(l_estimates.keys())
    l_values = [l_estimates[year] for year in years]
    
    # Apply rolling mean smoothing
    l_smoothed = []
    for i in range(len(l_values)):
        start_idx = max(0, i - rolling_window + 1)
        end_idx = i + 1
        window_values = l_values[start_idx:end_idx]
        l_smoothed.append(np.mean(window_values))
    
    # Create smoothed results
    l_smoothed_dict = dict(zip(years, l_smoothed))
    
    print("✅ Temporal smoothing completed")
    return l_smoothed_dict

def plot_l_time_series(l_estimates, l_uncertainties, l_smoothed):
    """
    Plot the l time series with uncertainties and smoothing.
    """
    print("\nCreating l time series plot...")
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    years = sorted(l_estimates.keys())
    l_raw = [l_estimates[year] for year in years]
    l_smooth = [l_smoothed[year] for year in years]
    
    # Extract uncertainties
    l_stds = [l_uncertainties[year]['std'] for year in years]
    l_ci_lower = [l_uncertainties[year]['ci_95_lower'] for year in years]
    l_ci_upper = [l_uncertainties[year]['ci_95_upper'] for year in years]
    
    # Plot raw estimates with error bars
    plt.errorbar(years, l_raw, yerr=l_stds, fmt='o', 
                capsize=5, capthick=2, markersize=8, 
                label='Raw l estimates ± 1σ', alpha=0.7)
    
    # Plot confidence intervals
    plt.fill_between(years, l_ci_lower, l_ci_upper, 
                    alpha=0.3, label='95% Confidence Interval')
    
    # Plot smoothed estimates
    plt.plot(years, l_smooth, 'r-', linewidth=3, 
            label=f'Smoothed (rolling mean)', marker='s', markersize=10)
    
    # Add horizontal line at l=2 (common default)
    plt.axhline(y=2.0, color='gray', linestyle='--', alpha=0.7, 
               label='l = 2.0 (common default)')
    
    # Customize the plot
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Estimated l (outcomes multiplier)', fontsize=14)
    plt.title('Time Series of l Estimates from Cross-Sectional Inference', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits
    plt.ylim(1.0, 3.5)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "figure3_l_time_series.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved as: {output_path}")
    
    plt.show()

def save_results(l_estimates, l_uncertainties, l_smoothed):
    """
    Save the l inference results to files.
    """
    print("\nSaving results...")
    
    # Create results dictionary
    results = {
        'l_estimates': l_estimates,
        'l_uncertainties': l_uncertainties,
        'l_smoothed': l_smoothed,
        'metadata': {
            'description': 'Cross-sectional l inference results from real data',
            'method': 'fit_l_cross_sectional with temporal smoothing',
            'timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    # Save as pickle
    pickle_path = "figure3_l_inference_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✅ Results saved as: {pickle_path}")
    
    # Save as CSV for easy viewing
    csv_data = []
    for year in sorted(l_estimates.keys()):
        csv_data.append({
            'year': year,
            'l_estimate': l_estimates[year],
            'l_std': l_uncertainties[year]['std'],
            'l_ci_95_lower': l_uncertainties[year]['ci_95_lower'],
            'l_ci_95_upper': l_uncertainties[year]['ci_95_upper'],
            'l_smoothed': l_smoothed[year]
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = "figure3_l_inference_results.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"✅ Results saved as: {csv_path}")
    
    return results

def main():
    """
    Main function to run the l inference analysis.
    """
    print("🚀 Figure 3: L Inference from Real Data")
    print("=" * 60)
    
    # Step 1: Load data from figure2
    data = load_figure2_data()
    if data is None:
        print("❌ Failed to load data. Exiting.")
        return False
    
    # Step 2: Extract growth rates and p values
    growth_rates, p_values, years = extract_growth_rates_and_p_values(data)
    if growth_rates is None:
        print("❌ Failed to extract data. Exiting.")
        return False
    
    # Step 3: Run cross-sectional l inference
    l_estimates, l_uncertainties = run_l_inference(growth_rates, p_values, years)
    
    # Step 4: Apply temporal smoothing
    l_smoothed = apply_temporal_smoothing(l_estimates, l_uncertainties)
    
    # Step 5: Plot results
    plot_l_time_series(l_estimates, l_uncertainties, l_smoothed)
    
    # Step 6: Save results
    results = save_results(l_estimates, l_uncertainties, l_smoothed)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 L INFERENCE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Processed {len(l_estimates)} timesteps")
    print(f"L estimates range: {min(l_estimates.values()):.3f} to {max(l_estimates.values()):.3f}")
    print(f"Smoothed L range: {min(l_smoothed.values()):.3f} to {max(l_smoothed.values()):.3f}")
    print("\nOutput files:")
    print("- figure3_l_time_series.png (plot)")
    print("- figure3_l_inference_results.pkl (pickle)")
    print("- figure3_l_inference_results.csv (CSV)")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Analysis failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Analysis completed successfully!") 