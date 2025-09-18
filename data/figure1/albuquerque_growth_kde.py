#!/usr/bin/env python3
"""
Albuquerque Growth Rate KDE Analysis
====================================

This script loads Albuquerque, NM income data and creates a KDE plot of growth rates
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
from scipy.stats import gaussian_kde

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

def create_growth_rate_kde(albuquerque_data, target_year=2019, save_path="figure1"):
    """
    Create KDE plot of growth rates for a specific year.
    
    Parameters:
    -----------
    albuquerque_data : dict
        Loaded Albuquerque, NM data
    target_year : int
        Year to analyze (default: 2019)
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
    
    # Find the target year in the growth rates
    years = sorted(albuquerque_data['year'].unique())
    if target_year not in years:
        print(f"Warning: {target_year} not found in data. Available years: {years}")
        # Use the middle year if target year not found
        target_year = years[len(years)//2]
        print(f"Using {target_year} instead")
    
    # Get the index of the target year in growth rates
    year_idx = years.index(target_year)
    if year_idx >= len(years) - 1:
        print(f"Warning: {target_year} is the last year, no growth rate available")
        year_idx = len(years) - 2  # Use second to last year
    
    # Extract growth rates for the target year
    growth_rates_target = growth_rates[:, year_idx]
    
    # Remove any remaining NaN values
    growth_rates_clean = growth_rates_target[~np.isnan(growth_rates_target)]
    
    print(f"Growth rates for {target_year}:")
    print(f"  Number of block groups: {len(growth_rates_clean)}")
    print(f"  Range: [{growth_rates_clean.min():.4f}, {growth_rates_clean.max():.4f}]")
    print(f"  Mean: {np.mean(growth_rates_clean):.4f}")
    print(f"  Std: {np.std(growth_rates_clean):.4f}")
    print(f"  Positive growth: {np.sum(growth_rates_clean > 0)} ({np.mean(growth_rates_clean > 0)*100:.1f}%)")
    print(f"  Negative growth: {np.sum(growth_rates_clean < 0)} ({np.mean(growth_rates_clean < 0)*100:.1f}%)")
    
    # Create the figure
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Custom color spectrum
    custom_colors = ['#172741', '#5F8089', '#C2C4B8', '#C3A983', '#483326']
    
    # Plot: Histogram with KDE overlay, colored by growth rate sign
    print("Creating histogram and KDE plot colored by growth rate sign...")
    
    # Separate positive and negative growth rates
    positive_mask = growth_rates_clean > 0
    negative_mask = growth_rates_clean < 0
    
    positive_growth = growth_rates_clean[positive_mask]
    negative_growth = growth_rates_clean[negative_mask]
    
    # Create histogram with different colors for positive and negative
    n_bins = min(30, int(np.sqrt(len(growth_rates_clean))))
    
    # Plot negative growth rates (low end of colorbar) first (behind)
    if len(negative_growth) > 0:
        ax1.hist(negative_growth, bins=n_bins, alpha=0.7, 
                color='#172741', edgecolor='#172741', linewidth=1, density=True, 
                label=f'Negative Growth ({len(negative_growth)} block groups)')
    
    # Plot positive growth rates (high end of colorbar) on top
    if len(positive_growth) > 0:
        ax1.hist(positive_growth, bins=n_bins, alpha=0.7, 
                color='#483326', edgecolor='#483326', linewidth=1, density=True,
                label=f'Positive Growth ({len(positive_growth)} block groups)')
    
    # Calculate and plot KDE for the full dataset
    kde = gaussian_kde(growth_rates_clean)
    x_range = np.linspace(growth_rates_clean.min(), growth_rates_clean.max(), 200)
    kde_values = kde(x_range)
    
    # Plot KDE
    ax1.plot(x_range, kde_values, color='#C3A983', linewidth=3, label='Full Dataset KDE')
    
    # Add vertical line for zero growth
    ax1.axvline(0, color='#C2C4B8', linestyle='--', linewidth=2, alpha=0.8, label='Zero Growth')
    
    # Add mean and median lines
    mean_growth = np.mean(growth_rates_clean)
    median_growth = np.median(growth_rates_clean)
    ax1.axvline(mean_growth, color='#5F8089', linestyle=':', linewidth=2, 
                label=f'Mean: {mean_growth:.4f}')
    ax1.axvline(median_growth, color='#C2C4B8', linestyle='-.', linewidth=2, 
                label=f'Median: {median_growth:.4f}')
    
    # Set labels and title
    ax1.set_xlabel('Log Growth Rate', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax1.set_title(f'Distribution of Growth Rates in {target_year}\nAlbuquerque, NM - Colored by Growth Sign', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    
    # Add statistics text
    stats_text = f'Total Block Groups: {len(growth_rates_clean)}\nPositive: {np.sum(positive_mask)} ({np.mean(positive_mask)*100:.1f}%)\nNegative: {np.sum(negative_mask)} ({np.mean(negative_mask)*100:.1f}%)'
    fig.text(0.02, 0.02, stats_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    figure_path = os.path.join(save_path, f'albuquerque_growth_kde_{target_year}.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {figure_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(save_path, f'albuquerque_growth_kde_{target_year}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Figure PDF saved to: {pdf_path}")
    
    return fig

def main():
    """Main function to create Albuquerque growth rate KDE analysis."""
    print("=" * 60)
    print("ALBUQUERQUE GROWTH RATE KDE ANALYSIS")
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
    
    # Create growth rate KDE analysis
    print("\nCreating growth rate KDE analysis for 2019...")
    fig = create_growth_rate_kde(albuquerque_data, target_year=2019)
    
    print("\n" + "=" * 60)
    print("ALBUQUERQUE GROWTH RATE KDE ANALYSIS COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - figure1/albuquerque_growth_kde_2019.png")
    print("  - figure1/albuquerque_growth_kde_2019.pdf")
    print("\nThe figure shows:")
    print("  - Histogram with KDE overlay of 2019 growth rates")
    print("  - Positive growth rates: Dark brown (high end of colorbar)")
    print("  - Negative growth rates: Dark blue (low end of colorbar)")
    print("  - Full dataset KDE overlay in gold")

if __name__ == "__main__":
    main() 