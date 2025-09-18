#!/usr/bin/env python3
"""
Chicago Growth Rate KDE Analysis
====================================

This script loads Chicago, IL income data and creates a KDE plot of growth rates
for 2019, with positive growth rates colored at the high end of the colorbar and
negative growth rates colored at the low end.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from pathlib import Path
from scipy.stats import gaussian_kde, t
from scipy.optimize import minimize

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'ytick.labelsize': 16,
    'xtick.labelsize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_Chicago_data(data_path):
    """
    Load Chicago, IL income data from the processed ACS data pickle file.
    
    Parameters:
    -----------
    data_path : str
        Path to the processed ACS data pickle file
        
    Returns:
    --------
    dict
        Loaded Chicago data with income trajectories, metadata, and error bars
    """
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded processed ACS data from {data_path}")
        
        # Find Chicago, IL data
        Chicago_data = None
        for cbsa_name, cbsa_data in data.items():
            if 'chicago-naperville-elgin' in cbsa_name.lower() and 'il-in' in cbsa_name.lower():
                Chicago_data = cbsa_data
                print(f"✓ Found Chicago, IL data: {cbsa_name}")
                break
        
        if Chicago_data is None:
            print("✗ Could not find Chicago, IL data in the file")
            print("Available CBSAs:")
            for cbsa_name in data.keys():
                print(f"  - {cbsa_name}")
            return None
        
        return Chicago_data
        
    except Exception as e:
        print(f"✗ Error loading processed ACS data: {e}")
        return None

def calculate_growth_rates(resources):
    """
    Calculate growth rates from resource trajectories.
    
    Parameters:
    -----------
    resources : np.ndarray
        Array of shape (n_block_groups, n_years) with income values
        
    Returns:
    --------
    np.ndarray
        Array of shape (n_block_groups, n_years-1) with growth rates
    """
    # Calculate growth rates: log difference between consecutive years
    growth_rates = np.diff(np.log(resources), axis=1)
    return growth_rates

def create_growth_rate_histogram(Chicago_data, target_year=2019, save_path="figure2", ax=None):
    """
    Create histogram of growth rates for a specific year.
    
    Parameters:
    -----------
    Chicago_data : dict
        Chicago data dictionary
    target_year : int
        Target year for the histogram
    save_path : str
        Directory to save the figure
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure.
        
    Returns:
    --------
    matplotlib.figure.Figure or None
        Figure object if ax is None, otherwise None
    """
    print(f"Creating growth rate histogram for {target_year}...")
    
    # Extract growth rates from processed data
    if 'detailed_growth_results' in Chicago_data:
        # Use processed data
        year_data = None
        for yd in Chicago_data['detailed_growth_results']:
            if yd['year'] == target_year:
                year_data = yd
                break
        
        if year_data is not None:
            growth_data = year_data['growth_data']
            growth_rates = growth_data['growth_rate'].values
            populations = growth_data['population'].values
            
            print(f"  Found {len(growth_rates)} growth rates for {target_year}")
            print(f"  Growth rates range: {growth_rates.min():.3f} to {growth_rates.max():.3f}")
            
            # Filter out invalid values
            valid_mask = np.isfinite(growth_rates)
            growth_rates_clean = growth_rates[valid_mask]
            populations_clean = populations[valid_mask]
            
            print(f"  Valid growth rates: {len(growth_rates_clean)}")
            
        else:
            print(f"  No data found for year {target_year}")
            return None
    else:
        print("  No detailed growth results found, cannot create histogram")
        return None
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        standalone = True
    else:
        standalone = False
    
    # Create histogram with population weights
    if len(populations_clean) > 0:
        # Normalize weights to sum to number of observations
        weights = populations_clean / np.sum(populations_clean) * len(populations_clean)
        
        # Create histogram
        n, bins, patches = ax.hist(growth_rates_clean, bins=30, alpha=0.7, 
                                   color='#C3A983', edgecolor='#483326', 
                                   linewidth=1, weights=weights, density=True)
        
        # Color the histogram bars based on growth rate sign
        # Create a custom colormap from low (negative) to high (positive)
        # Use the same color scheme as other figures: dark blue to dark brown
        custom_colors = ['#172741', '#5F8089', '#C2C4B8', '#C3A983', '#483326']
        colors = plt.cm.colors.LinearSegmentedColormap.from_list('custom', custom_colors)(np.linspace(0, 1, len(patches)))
        
        # Color each patch based on whether the bin center is positive or negative
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center < 0:
                # Negative growth rates: use low end of colorbar (dark blue)
                patch.set_facecolor(colors[0])
            else:
                # Positive growth rates: use high end of colorbar (dark brown)
                patch.set_facecolor(colors[-1])
        
        # Add vertical line for zero growth
        ax.axvline(0, color='#C2C4B8', linestyle='--', linewidth=2, alpha=0.8, label='Zero Growth')
        
        # Add mean and median lines
        mean_growth = np.mean(growth_rates_clean)
        median_growth = np.median(growth_rates_clean)
        ax.axvline(mean_growth, color='#5F8089', linestyle=':', linewidth=2, 
                   label=f'Mean: {mean_growth:.4f}')
        ax.axvline(median_growth, color='#5F8089', linestyle=':', linewidth=2, 
                   label=f'Median: {median_growth:.4f}')
        
        # Set labels and title
        # ax.set_xlabel('Log Growth Rate', fontsize=14, fontweight='bold')
        # ax.set_ylabel('Density', fontsize=14, fontweight='bold')
        # ax.set_title(f'Growth Rate Distribution - {target_year}\nChicago, IL', 
        #              fontsize=16, fontweight='bold')
        
        # Set x-axis limits
        ax.set_xlim(-0.5, 0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        # ax.legend(fontsize=10)
        
        print(f"  Mean growth rate: {mean_growth:.4f}")
        print(f"  Median growth rate: {median_growth:.4f}")
        print(f"  Standard deviation: {np.std(growth_rates_clean):.4f}")
    
    # Save the figure if standalone mode
    if standalone:
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(save_path, exist_ok=True)
        
        # Save as PDF for publication
        pdf_path = os.path.join(save_path, f'Chicago_growth_histogram_{target_year}.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"✓ Figure PDF saved to: {pdf_path}")
        
        return fig
    else:
        # Just return None if plotting on provided axis
        return None

def create_p_time_series(Chicago_data, save_path="figure2", ax=None):
    """
    Create p time series plot with error bars.
    
    Parameters:
    -----------
    Chicago_data : dict
        Chicago data dictionary
    save_path : str
        Directory to save the figure
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure.
        
    Returns:
    --------
    matplotlib.figure.Figure or None
        Figure object if ax is None, otherwise None
    """
    print("Creating p time series plot...")
    
    # Use processed ACS data with error bars
    if 'summary' in Chicago_data:
        print("Using processed ACS data with error bars...")
        
        summary_df = Chicago_data['summary']
        p_values = summary_df['p_value'].tolist()
        p_moes = summary_df['p_moe'].tolist()
        p_years = summary_df['year'].tolist()
        
        print(f"Loaded p values with error bars for years: {p_years}")
        print(f"P values: {[f'{p:.3f}' for p in p_values]}")
        print(f"P MOEs: {[f'{moe:.4f}' for moe in p_moes]}")
        
    else:
        print("Falling back to raw data calculation...")
        # Calculate p time series for all years (fallback method)
        years = sorted(Chicago_data['year'].unique())
        p_values = []
        p_years = []
        
        print(f"Calculating p time series for years: {years}")
        
        # We need to calculate growth rates between consecutive years
        for i in range(1, len(years)):
            current_year = years[i]
            previous_year = years[i-1]
            
            print(f"  Processing {previous_year} -> {current_year}")
            
            # Get data for both years
            current_data = Chicago_data[Chicago_data['year'] == current_year]
            previous_data = Chicago_data[Chicago_data['year'] == previous_year]
            
            print(f"    Current year data: {len(current_data)} records")
            print(f"    Previous year data: {len(previous_data)} records")
            
            if len(current_data) > 0 and len(previous_data) > 0:
                # Merge data by block group to calculate growth rates
                merged_data = current_data.merge(previous_data, on='block_group_fips', 
                                               suffixes=('_current', '_previous'))
                
                print(f"    Merged data: {len(merged_data)} records")
                
                if len(merged_data) > 0:
                    # Calculate growth rates
                    growth_rates = np.log(merged_data['mean_income_current'] / merged_data['mean_income_previous'])
                    
                    # Remove any infinite or NaN values
                    valid_mask = np.isfinite(growth_rates)
                    growth_rates_clean = growth_rates[valid_mask]
                    merged_data_clean = merged_data[valid_mask]
                    
                    print(f"    Valid growth rates: {len(growth_rates_clean)}")
                    
                    if len(growth_rates_clean) > 0:
                        # Calculate population-weighted p
                        total_pop = merged_data_clean['population_current'].sum()
                        positive_mask = growth_rates_clean > 0
                        positive_pop = merged_data_clean.loc[positive_mask, 'population_current'].sum()
                        
                        print(f"    Total population: {total_pop:,.0f}")
                        print(f"    Positive growth population: {positive_pop:,.0f}")
                        
                        if total_pop > 0:
                            print(f"    Positive growth population: {positive_pop:,.0f}")
                            print(f"    Negative growth population: {total_pop - positive_pop:,.0f}")
                            p = positive_pop / total_pop
                            p_values.append(p)
                            p_years.append(current_year)
                            print(f"    p = {p:.3f}")
                        else:
                            print(f"    Warning: Total population is 0")
                    else:
                        print(f"    Warning: No valid growth rates")
                else:
                    print(f"    Warning: No data after merge")
            else:
                print(f"    Warning: Missing data for one or both years")
        
        # For fallback method, we don't have error bars
        p_moes = [0.0] * len(p_values)
    
    print(f"Final p values: {p_values}")
    print(f"Final p years: {p_years}")
    print(f"Final p MOEs: {p_moes}")
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        standalone = True
    else:
        standalone = False
    
    # Plot p time series
    if len(p_values) > 0:
        # Convert to regular Python types to avoid numpy plotting issues
        p_years_clean = [int(year) for year in p_years]
        p_values_clean = [float(p) for p in p_values]
        p_moes_clean = [float(moe) for moe in p_moes]
        
        print(f"Plotting p values: {p_values_clean}")
        print(f"Plotting years: {p_years_clean}")
        print(f"Plotting MOEs: {p_moes_clean}")
        
        # Plot the time series with error bars
        ax.errorbar(p_years_clean, p_values_clean, yerr=p_moes_clean, 
                    color='#483326', linewidth=3, marker='o', markersize=6,
                    capsize=5, capthick=2, elinewidth=2, label='p values with error bars')
        
        # Add horizontal line at p = 0.5
        ax.axhline(0.5, color='#3D3D3D', linestyle='--', linewidth=2, alpha=0.7, label='p = 0.5 (Equal Growth/Loss)')
        
        # Add statistics
        mean_p = np.mean(p_values_clean)
        std_p = np.std(p_values_clean)
        
        # Set labels and title
        # ax.set_xlabel('Year', fontsize=14, fontweight='bold')
        # ax.set_ylabel('Population-Weighted p', fontsize=14, fontweight='bold')
        # ax.set_title('Time Series of p: Fraction with Positive Growth\nChicago, IL', 
        #              fontsize=16, fontweight='bold')
        
        # Set grid and limits
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(.3, 1)
        # ax.legend(fontsize=10)
        
        print(f"  Mean p: {mean_p:.3f}")
        print(f"  Std p: {std_p:.3f}")
    
    # Save the figure if standalone mode
    if standalone:
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(save_path, exist_ok=True)
        
        # Save as PDF for publication
        pdf_path = os.path.join(save_path, 'Chicago_p_time_series.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"✓ Figure PDF saved to: {pdf_path}")
        
        return fig
    else:
        # Just return None if plotting on provided axis
        return None

def save_figure2_data(Chicago_data, growth_rates, p_values, p_years, save_path="figure2"):
    """
    Save the growth rates and p values data for use by figure3.py.
    
    Parameters:
    -----------
    Chicago_data : dict
        Chicago data dictionary
    growth_rates : np.ndarray
        Array of growth rates for 2015
    p_values : list
        List of p values for each year
    p_years : list
        List of years corresponding to p values
    save_path : str
        Directory to save the data files
    """
    print(f"\nSaving figure2 data to {save_path}...")
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'city': 'Chicago, IL',
        'description': 'Figure 2 data: Growth rates (2015) and p values for Chicago, IL',
        'years': p_years,
        'n_block_groups': len(growth_rates) if growth_rates is not None else 0,
        'n_years': len(p_years),
        'data_source': 'processed_acs_data.pkl'
    }
    
    # Check if we have error bars from processed data
    has_error_bars = 'summary' in Chicago_data
    
    if has_error_bars:
        print("Including error bars from processed ACS data...")
        summary_df = Chicago_data['summary']
        
        # Extract error bars for p values
        p_moes = summary_df['p_moe'].tolist()
        
        # Save with error bars
        data_to_save = {
            'growth_rates': growth_rates,
            'p_values': p_values,
            'p_moes': p_moes,
            'years': p_years,
            'metadata': metadata,
            'has_error_bars': True
        }
    else:
        print("No error bars available, saving basic data...")
        data_to_save = {
            'growth_rates': growth_rates,
            'p_values': p_values,
            'years': p_years,
            'metadata': metadata,
            'has_error_bars': False
        }
    
    # Save as pickle
    pickle_path = os.path.join(save_path, "figure2_data.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"✓ Figure2 data pickle saved to: {pickle_path}")
    
    # Save p values as CSV for easy inspection
    if has_error_bars:
        p_df = pd.DataFrame({
            'year': p_years,
            'p_value': p_values,
            'p_moe': p_moes
        })
    else:
        p_df = pd.DataFrame({
            'year': p_years,
            'p_value': p_values
        })
    
    csv_path = os.path.join(save_path, "figure2_p_values.csv")
    p_df.to_csv(csv_path, index=False)
    print(f"✓ P values CSV saved to: {csv_path}")
    
    # Print summary
    print(f"\nData summary:")
    print(f"  - Growth rates (2015): {len(growth_rates)} block groups")
    print(f"  - P values: {len(p_values)} years")
    print(f"  - Error bars: {'Yes' if has_error_bars else 'No'}")
    if has_error_bars:
        print(f"  - P value MOEs: {[f'{moe:.4f}' for moe in p_moes]}")

def main():
    """Main function to run the analysis."""
    print("=" * 60)
    print("Chicago GROWTH RATE ANALYSIS")
    print("=" * 60)
    
    # Data path - use processed ACS data with error bars
    data_path = "../../urbandata/data/processed_acs_data.pkl"
    
    # Create output directory
    save_path = "figure2"
    os.makedirs(save_path, exist_ok=True)
    
    # Load Chicago data
    Chicago_data = load_Chicago_data(data_path)
    if Chicago_data is None:
        print("❌ Failed to load Chicago data. Exiting.")
        return
    
    print(f"✅ Successfully loaded Chicago data")
    print(f"Data keys: {list(Chicago_data.keys())}")
    
    # Create the figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
    
    # Create growth rate histogram for 2015
    create_growth_rate_histogram(Chicago_data, target_year=2015, save_path=save_path, ax=ax1)
    
    # Create p time series plot
    create_p_time_series(Chicago_data, save_path=save_path, ax=ax2)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the combined figure
    fig_path = os.path.join(save_path, "figure2_combined.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined figure saved to: {fig_path}")
    
    # Also save as PNG
    fig_path_png = os.path.join(save_path, "figure2_combined.png")
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    print(f"✓ Combined figure PNG saved to: {fig_path_png}")
    
    # Save data for figure3.py - just the p values and years
    if 'summary' in Chicago_data:
        # Use processed data
        summary_df = Chicago_data['summary']
        p_values = summary_df['p_value'].tolist()
        p_years = summary_df['year'].tolist()
        
        # For figure3.py, we need growth rates in a format it can use
        # We'll create a simple array for 2015 (the year we analyzed)
        if 'detailed_growth_results' in Chicago_data:
            # Find 2015 data
            year_2015_data = None
            for yd in Chicago_data['detailed_growth_results']:
                if yd['year'] == 2015:
                    year_2015_data = yd
                    break
            
            if year_2015_data is not None:
                growth_data = year_2015_data['growth_data']
                growth_rates_2015 = growth_data['growth_rate'].values
                print(f"  Extracted {len(growth_rates_2015)} growth rates for 2015")
            else:
                growth_rates_2015 = np.array([])
                print("  No 2015 data found")
        else:
            growth_rates_2015 = np.array([])
        
    else:
        # Fallback to empty data
        p_values, p_years = [], []
        growth_rates_2015 = np.array([])
    
    # Save the data
    save_figure2_data(Chicago_data, growth_rates_2015, p_values, p_years)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 