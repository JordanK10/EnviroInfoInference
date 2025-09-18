#!/usr/bin/env python3
"""
Albuquerque P Time Series Analysis
==================================

This script loads Albuquerque, NM income data and calculates the time series of p,
where p is the fraction of block groups with positive growth rates in each year.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
from pathlib import Path

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 2.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 2.4,
    'ytick.major.width': 2.4,
    'ytick.labelsize': 16,
    'xtick.labelsize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_albuquerque_data(data_path):
    """
    Load Albuquerque, NM income data from the CBSA ACS data pickle file.
    
    Parameters:
    -----------
    data_path : str
        Path to the CBSA ACS data pickle file
        
    Returns:
    --------
    dict
        Loaded Albuquerque data with income trajectories and metadata
    """
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded CBSA ACS data from {data_path}")
        
        # Find Albuquerque, NM data
        albuquerque_data = None
        for cbsa_name, cbsa_data in data.items():
            if 'albuquerque' in cbsa_name.lower() and 'nm' in cbsa_name.lower():
                albuquerque_data = cbsa_data
                print(f"✓ Found Albuquerque, NM data: {cbsa_name}")
                break
        
        if albuquerque_data is None:
            print("✗ Could not find Albuquerque, NM data in the file")
            print("Available CBSAs:")
            for cbsa_name in data.keys():
                print(f"  - {cbsa_name}")
            return None
        
        return albuquerque_data
        
    except Exception as e:
        print(f"✗ Error loading CBSA ACS data: {e}")
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

def calculate_p_timeseries(albuquerque_data, save_path="figure2"):
    """
    Calculate and plot the time series of p (fraction of positive growth rates).
    
    Parameters:
    -----------
    albuquerque_data : dict
        Loaded Albuquerque, NM data
    save_path : str
        Directory to save the figure
    """
    # Handle the long-format DataFrame structure
    if isinstance(albuquerque_data, pd.DataFrame):
        print("Data is in long format DataFrame, reshaping to wide format...")
        
        # Check if we have the expected columns
        expected_cols = ['year', 'block_group_fips', 'mean_income', 'population']
        if not all(col in albuquerque_data.columns for col in expected_cols):
            raise ValueError(f"Expected columns {expected_cols}, got {list(albuquerque_data.columns)}")
        
        # Pivot the data to wide format: block_group_fips as rows, years as columns
        resources = albuquerque_data.pivot(index='block_group_fips', columns='year', values='mean_income')
        
        # Sort by year to ensure chronological order
        resources = resources.reindex(sorted(resources.columns), axis=1)
        
        print(f"Reshaped data: {resources.shape[0]} block groups × {resources.shape[1]} years")
        print(f"Year range: {list(resources.columns)}")
        
        # Store the block group identifiers for later use
        block_group_ids = resources.index.values
        
        print(f"Block group IDs shape: {block_group_ids.shape}")
        print(f"First few block group IDs: {block_group_ids[:5]}")
        
        # Convert to numpy array
        resources = resources.values
        
        # Handle missing values (NaN) by forward-filling within each block group
        print("Handling missing values...")
        n_missing_before = np.isnan(resources).sum()
        if n_missing_before > 0:
            print(f"Found {n_missing_before} missing values, applying forward-fill...")
            # Forward-fill missing values within each block group
            for i in range(resources.shape[0]):
                # Forward-fill, then backward-fill for any remaining NaNs at the beginning
                resources[i, :] = pd.Series(resources[i, :]).fillna(method='ffill').fillna(method='bfill').values
            n_missing_after = np.isnan(resources).sum()
            print(f"After forward-fill: {n_missing_after} missing values remaining")
            
            if n_missing_after > 0:
                print("Warning: Some missing values remain after forward-fill")
                # For any remaining NaNs, use the mean of that year across all block groups
                for j in range(resources.shape[1]):
                    year_mean = np.nanmean(resources[:, j])
                    if not np.isnan(year_mean):
                        mask = np.isnan(resources[:, j])
                        resources[mask, j] = year_mean
                        print(f"Filled {mask.sum()} remaining NaNs in year {j} with mean {year_mean:.0f}")
        
    else:
        raise ValueError("Data must be a pandas DataFrame")
    
    # Calculate growth rates
    growth_rates = calculate_growth_rates(resources)
    
    # Get years for growth rates (one fewer than income years)
    years = sorted(albuquerque_data['year'].unique())
    growth_years = years[:-1]  # Growth rates are between consecutive years
    
    print(f"Growth rates calculated for {len(growth_years)} time periods:")
    print(f"Years: {growth_years}")
    
    # Calculate p for each time period (population-weighted)
    p_values = []
    p_unweighted = []  # For comparison
    n_positive = []
    n_total = []
    total_population = []
    
    for i in range(growth_rates.shape[1]):
        growth_rates_period = growth_rates[:, i]
        
        # Get population data for this time period
        # Note: population data is available for each year, so we need to handle the growth rate period
        # For growth rates between years t and t+1, we'll use population from year t
        if i < resources.shape[1] - 1:  # Make sure we don't go out of bounds
            year_data = albuquerque_data[albuquerque_data['year'] == years[i]]
        else:
            # For the last growth rate period, use the second-to-last year's population
            year_data = albuquerque_data[albuquerque_data['year'] == years[-2]]
        
        print(f"    Year {years[i] if i < resources.shape[1] - 1 else years[-2]}: {len(year_data)} records")
        print(f"    Year data columns: {list(year_data.columns)}")
        
        # Create mask for valid growth rates (no NaN)
        valid_mask = ~np.isnan(growth_rates_period)
        growth_rates_clean = growth_rates_period[valid_mask]
        
        # Get the block group indices that have valid growth rates
        valid_indices = np.where(valid_mask)[0]
        
        # Get population data for the same block groups that have valid growth rates
        # We need to match by the block group IDs, not by position
        population_clean = []
        for idx in valid_indices:
            # Get the block group ID for this index
            block_group_id = block_group_ids[idx]
            
            # Find the population for this specific block group in the year data
            block_group_pop = year_data[year_data['block_group_fips'] == block_group_id]['population'].values
            
            if len(block_group_pop) > 0:
                population_clean.append(block_group_pop[0])
            else:
                # If no population data found, use a default value or skip
                print(f"    Warning: No population data for block group {block_group_id}")
                population_clean.append(0)  # Default to 0 population
        
        # Convert to numpy array
        population_clean = np.array(population_clean)
        
        # Ensure we have the same number of valid growth rates and population values
        if len(population_clean) != len(growth_rates_clean):
            print(f"    Warning: Mismatch in data lengths. Growth rates: {len(growth_rates_clean)}, Population: {len(population_clean)}")
            # This shouldn't happen now, but just in case
            min_len = min(len(growth_rates_clean), len(population_clean))
            growth_rates_clean = growth_rates_clean[:min_len]
            population_clean = population_clean[:min_len]
        
        if len(growth_rates_clean) > 0:
            # Calculate unweighted p (original method)
            n_pos_unweighted = np.sum(growth_rates_clean > 0)
            n_tot_unweighted = len(growth_rates_clean)
            p_unw = n_pos_unweighted / n_tot_unweighted if n_tot_unweighted > 0 else 0
            
            # Calculate population-weighted p
            positive_mask = growth_rates_clean > 0
            total_pop = np.sum(population_clean)
            positive_pop = np.sum(population_clean[positive_mask])
            p_weighted = positive_pop / total_pop if total_pop > 0 else 0
            
            p_values.append(p_weighted)
            p_unweighted.append(p_unw)
            n_positive.append(n_pos_unweighted)
            n_total.append(n_tot_unweighted)
            total_population.append(total_pop)
            
            print(f"  {growth_years[i]}: p_weighted = {p_weighted:.3f}, p_unweighted = {p_unw:.3f}")
            print(f"    Population: {total_pop:,.0f}, Positive: {positive_pop:,.0f} ({positive_pop/total_pop*100:.1f}%)")
        else:
            p_values.append(0)
            p_unweighted.append(0)
            n_positive.append(0)
            n_total.append(0)
            total_population.append(0)
            print(f"  {growth_years[i]}: No valid data")
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Custom color spectrum
    custom_colors = ['#172741', '#5F8089', '#C2C4B8', '#C3A983', '#483326']
    
    # Plot 1: P time series (weighted vs unweighted)
    print("Creating p time series plot...")
    
    # Plot both weighted and unweighted p values
    ax1.plot(growth_years, p_values, color='#483326', linewidth=3, marker='o', markersize=6, 
             label='Population-Weighted p')
    ax1.plot(growth_years, p_unweighted, color='#5F8089', linewidth=2, marker='s', markersize=4, 
             linestyle='--', label='Unweighted p')
    
    ax1.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Fraction of Positive Growth (p)', fontsize=14, fontweight='bold')
    ax1.set_title('Time Series of p: Fraction with Positive Growth\nAlbuquerque, NM (Population-Weighted vs Unweighted)', 
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    # Add horizontal line at p = 0.5
    ax1.axhline(0.5, color='#C2C4B8', linestyle='--', linewidth=2, alpha=0.7, label='p = 0.5 (Equal Growth/Loss)')
    
    # Add statistics text
    mean_p_weighted = np.mean(p_values)
    std_p_weighted = np.std(p_values)
    mean_p_unweighted = np.mean(p_unweighted)
    std_p_unweighted = np.std(p_unweighted)
    
    stats_text = f'Weighted: μ={mean_p_weighted:.3f}, σ={std_p_weighted:.3f}\nUnweighted: μ={mean_p_unweighted:.3f}, σ={std_p_unweighted:.3f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Count of positive vs negative growth rates
    print("Creating count comparison plot...")
    
    x_pos = np.arange(len(growth_years))
    width = 0.35
    
    # Plot positive and negative counts
    negative_counts = [n_tot - n_pos for n_pos, n_tot in zip(n_positive, n_total)]
    
    ax2.bar(x_pos - width/2, n_positive, width, label='Positive Growth', 
            color='#483326', alpha=0.8)
    ax2.bar(x_pos + width/2, negative_counts, width, label='Negative Growth', 
            color='#172741', alpha=0.8)
    
    ax2.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Block Groups', fontsize=14, fontweight='bold')
    ax2.set_title('Count of Block Groups by Growth Sign Over Time\nAlbuquerque, NM', 
                  fontsize=16, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(growth_years)
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add total counts text
    total_text = f'Total Block Groups: {n_total[0] if n_total else 0}'
    ax2.text(0.02, 0.98, total_text, transform=ax2.transAxes, 
             verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    
    # Save as PNG
    figure_path = os.path.join(save_path, 'abq_p_timeseries.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {figure_path}")
    
    # Save as PDF
    pdf_path = os.path.join(save_path, 'abq_p_timeseries.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Figure PDF saved to: {pdf_path}")
    
    # Save p values to CSV
    csv_path = os.path.join(save_path, 'abq_p_timeseries.csv')
    p_df = pd.DataFrame({
        'year': growth_years,
        'p_weighted': p_values,
        'p_unweighted': p_unweighted,
        'n_positive': n_positive,
        'n_negative': [n_tot - n_pos for n_pos, n_tot in zip(n_positive, n_total)],
        'n_total': n_total,
        'total_population': total_population
    })
    p_df.to_csv(csv_path, index=False)
    print(f"✓ Data saved to: {csv_path}")
    
    return fig, p_values, growth_years

def main():
    """Main function to create Albuquerque p time series analysis."""
    print("=" * 60)
    print("ALBUQUERQUE P TIME SERIES ANALYSIS")
    print("=" * 60)
    
    # Path to CBSA ACS data
    data_path = "figure1/cbsa_acs_data.pkl"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        print("Please ensure cbsa_acs_data.pkl is in the figure1 directory.")
        return
    
    # Load Albuquerque data
    albuquerque_data = load_albuquerque_data(data_path)
    if albuquerque_data is None:
        return
    
    # Create p time series analysis
    print("\nCreating p time series analysis...")
    fig, p_values, years = calculate_p_timeseries(albuquerque_data)
    
    print("\n" + "=" * 60)
    print("ALBUQUERQUE P TIME SERIES ANALYSIS COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - figure2/abq_p_timeseries.png")
    print("  - figure2/abq_p_timeseries.pdf")
    print("  - figure2/abq_p_timeseries.csv")
    print("\nThe figure shows:")
    print("  - Top: Time series of p (fraction of positive growth rates)")
    print("  - Bottom: Count of positive vs negative growth rates over time")
    print("  - Horizontal line at p = 0.5 for reference")
    print(f"\nSummary statistics:")
    print(f"  Mean p: {np.mean(p_values):.3f}")
    print(f"  Std p: {np.std(p_values):.3f}")
    print(f"  Range: [{min(p_values):.3f}, {max(p_values):.3f}]")

if __name__ == "__main__":
    main() 