#!/usr/bin/env python3
"""
Figure 1A: Log Incomes vs Growth Rates for Albuquerque, NM
==========================================================

This script creates Figure 1A for the paper, showing the relationship between
log incomes and growth rates from real Albuquerque, NM ACS data.

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
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
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
        Array of shape (n_block_groups, n_timesteps + 1) with resource values
        
    Returns:
    --------
    np.ndarray
        Array of shape (n_block_groups, n_timesteps) with growth rates
    """
    # Calculate growth rates: (r_t - r_{t-1}) / r_{t-1}
    growth_rates = np.diff(np.log(resources), axis=1) 
    return growth_rates

def create_figure1a(albuquerque_data, save_path="figure1"):
    """
    Create Figure 1A: Time Series of Income vs Growth Rates for Albuquerque, NM.
    
    Parameters:
    -----------
    albuquerque_data : dict
        Loaded Albuquerque, NM data
    save_path : str
        Directory to save the figure
    """
    # Extract income data - structure may vary, let's explore it
    print("Data structure:")
    for key, value in albuquerque_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
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
        # Try to find income data in different possible structures (fallback for other formats)
        resources = None
        if 'income_data' in albuquerque_data:
            resources = albuquerque_data['income_data']
            print("Found income data in 'income_data' key")
        elif 'block_group_income' in albuquerque_data:
            resources = albuquerque_data['block_group_income']
            print("Found income data in 'block_group_income' key")
        elif 'income_trajectories' in albuquerque_data:
            resources = albuquerque_data['income_trajectories']
            print("Found income data in 'income_trajectories' key")
        else:
            # Look for any array that could be income data
            for key, value in albuquerque_data.items():
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    resources = value
                    print(f"Using {key} as income data (fallback)")
                    break
            
            if resources is None:
                # Try to find any 2D array that might be income data
                for key, value in albuquerque_data.items():
                    if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] > 1:
                        resources = value
                        print(f"Using {key} as income data (shape: {value.shape})")
                        break
            
            if resources is None:
                raise ValueError("Could not find income data in the expected format. Available keys: " + 
                               str(list(albuquerque_data.keys())))
    
    # Ensure resources is 2D array (n_block_groups, n_years)
    if resources.ndim == 1:
        resources = resources.reshape(1, -1)
    
    n_block_groups, n_timesteps_plus_1 = resources.shape
    n_timesteps = n_timesteps_plus_1 - 1
    
    print(f"Data shape: {n_block_groups} block groups, {n_timesteps} timesteps")
    
    # Calculate growth rates
    growth_rates = calculate_growth_rates(resources)  # Shape: (n_block_groups, n_timesteps)
    
    # Calculate log incomes (using resources at each timestep)
    log_incomes = np.log(resources) # Shape: (n_block_groups, n_timesteps + 1)
    
    # Calculate cumulative growth rates for coloring
    cumulative_growth_rates = np.zeros(n_block_groups)
    for block_group_id in range(n_block_groups):
        # Calculate total growth from start to end
        initial_resources = resources[block_group_id, 0]
        final_resources = resources[block_group_id, -1]
        cumulative_growth_rates[block_group_id] = (np.log(final_resources) - np.log(initial_resources))
    
    # Convert to relative growth rate (percentile rank) for coloring
    from scipy.stats import rankdata
    relative_growth_ranks = rankdata(cumulative_growth_rates) / len(cumulative_growth_rates)  # 0 to 1
    
    # Sort block groups by cumulative growth rate for consistent coloring
    sorted_indices = np.argsort(cumulative_growth_rates)
    
    print(f"Coloring {n_block_groups} block groups by cumulative growth rate order")
    
    # Create the figure with three subplots: two time series and one histogram
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create time array for x-axis - use actual years if available
    if isinstance(albuquerque_data, pd.DataFrame):
        # Extract years from the original data
        years = sorted(albuquerque_data['year'].unique())
        time_steps = years
        time_steps_growth = years[1:]  # Growth rates start from second year
        print(f"Using actual years: {years}")
    else:
        # Fallback to generic timesteps
        time_steps = np.arange(n_timesteps + 1)
        time_steps_growth = np.arange(1, n_timesteps + 1)  # Growth rates start from timestep 1
    
    # Custom color spectrum
    custom_colors1 = ['#172741', '#5F8089', '#C2C4B8', '#C3A983', '#483326']
    
    # Plot 1: Time series of log incomes for all block groups
    print("Plotting log income time series...")
    print(f"Cumulative growth rates range: {cumulative_growth_rates.min():.3f} to {cumulative_growth_rates.max():.3f}")
    print(f"Using relative growth ranks (percentiles) for coloring")
    
    # Calculate empirical confidence intervals based on actual data
    print("Calculating empirical confidence intervals...")
    
    # Calculate empirical mean and standard deviation at each timestep
    empirical_mean = np.mean(log_incomes, axis=0)
    empirical_std = np.std(log_incomes, axis=0)
    
    # Calculate 95% CI bounds around the empirical mean trajectory
    ci_lower = empirical_mean - 1.96 * empirical_std 
    ci_upper = empirical_mean + 1.96 * empirical_std 
    
    # # Plot the empirical confidence interval
    # ax1.fill_between(time_steps, ci_lower, ci_upper, 
    #                  color='#3D3D3D', alpha=0.5, label='Empirical 95% CI (Data)', zorder=100)
    
    # Plot the empirical mean trajectory
    ax1.plot(time_steps, empirical_mean, color='#3D3D3D', linewidth=2, 
             linestyle='--', alpha=0.8, label='Empirical Mean Trajectory')
    
    # Plot individual trajectories colored by growth rate order
    print("Plotting individual trajectories...")
    
    # Plot all trajectories using the custom color spectrum
    for i, block_group_id in enumerate(sorted_indices):
        # Get color based on growth rate rank
        color_idx = int(relative_growth_ranks[block_group_id] * (len(custom_colors1) - 1))
        color = custom_colors1[color_idx]
        
        ax1.plot(time_steps, log_incomes[block_group_id, :], 
                color=color, alpha=0.6, linewidth=0.8)
    
    # Plot the overall mean trajectory
    overall_mean = np.mean(log_incomes, axis=0)
    ax1.plot(time_steps, overall_mean, color='#3D3D3D', linewidth=3, 
             linestyle='--', alpha=0.8, label='Overall Mean Trajectory')
    
    # Plot 2: Time series of growth rates for all block groups
    print("Plotting growth rate time series...")
    
    # Plot growth rates colored by growth rate order
    for i, block_group_id in enumerate(sorted_indices):
        # Get color based on growth rate rank
        color_idx = int(relative_growth_ranks[block_group_id] * (len(custom_colors1) - 1))
        color = custom_colors1[color_idx]
        
        ax2.plot(time_steps_growth, growth_rates[block_group_id, :], 
                color=color, alpha=0.6, linewidth=0.8)
    
    # Plot mean growth rate trajectory
    overall_growth_mean = np.mean(growth_rates, axis=0)
    ax2.plot(time_steps_growth, overall_growth_mean, color='#3D3D3D', linewidth=3, 
             linestyle='--', alpha=0.8, label='Overall Mean Growth Rate')
    
    # Plot 3: Histogram of growth rates for a specific year
    print("Plotting growth rate histogram...")
    
    # Choose a year to histogram (middle year to avoid edge effects)
    hist_year_idx = len(time_steps_growth) // 2
    hist_year = time_steps_growth[hist_year_idx]
    growth_rates_hist = growth_rates[:, hist_year_idx]
    
    print(f"Histogramming growth rates for year {hist_year} (index {hist_year_idx})")
    print(f"Growth rates range: {growth_rates_hist.min():.4f} to {growth_rates_hist.max():.4f}")
    print(f"Mean growth rate: {np.mean(growth_rates_hist):.4f}")
    print(f"Std growth rate: {np.std(growth_rates_hist):.4f}")
    
    # Create histogram with custom colors
    n_bins = min(30, int(np.sqrt(len(growth_rates_hist))))  # Sturges' rule
    hist, bin_edges, _ = ax3.hist(growth_rates_hist, bins=50, alpha=0.7, 
                                  color='#5F8089', edgecolor='#172741', linewidth=1)
    
    # # Add vertical line for mean
    # mean_growth = np.mean(growth_rates_hist)
    # ax3.axvline(mean_growth, color='#3D3D3D', linestyle='--', linewidth=2, 
    #             label=f'Mean: {mean_growth:.4f}')
    
    # # Add vertical line for median
    # median_growth = np.median(growth_rates_hist)
    # ax3.axvline(median_growth, color='#C3A983', linestyle=':', linewidth=2, 
    #             label=f'Median: {median_growth:.4f}')
    
    # Add vertical line for zero (no growth)
    ax3.axvline(0, color='#483326', linestyle='-', linewidth=1, alpha=0.7, 
                label='Zero Growth')
    
    # Add text box with statistics
    # stats_text = f'Mean: {mean_growth:.4f}\nStd: {np.std(growth_rates_hist):.4f}\nN: {len(growth_rates_hist)}'
    # ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
    #          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set histogram properties
    ax3.set_xlabel('Log Growth Rate', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax3.set_xlim(-.4,.4)
    ax3.set_title(f'Distribution of Growth Rates\nYear {hist_year}', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend()
    
    # Add colorbar to show growth rate mapping
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    
    cmap = LinearSegmentedColormap.from_list('custom', custom_colors1)
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # cbar = plt.colorbar(sm, ax=[ax1, ax2, ax3], shrink=0.8, aspect=30)
    # cbar.set_label('Cumulative Growth Rate Rank (Percentile)', fontsize=12, fontweight='bold')
    # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    # cbar.set_ticklabels(['Lowest', '25%', '50%', '75%', 'Highest'])
    
    # Set labels and titles
    # ax1.set_xlabel('Year', fontsize=14, fontweight='bold')
    # ax1.set_ylabel('Log Income (log dollars)', fontsize=14, fontweight='bold')
    # ax1.set_title('Time Series of Log Incomes\n(Colored by Cumulative Growth Rate)', fontsize=16, fontweight='bold')
    # ax1.grid(True, alpha=0.3, linestyle='--')
    # ax1.legend()
    
    # ax2.set_xlabel('Year', fontsize=14, fontweight='bold')
    # ax2.set_ylabel('Growth Rate (fractional change)', fontsize=14, fontweight='bold')
    # ax2.set_title('Time Series of Growth Rates\n(Colored by Cumulative Growth Rate)', fontsize=16, fontweight='bold')
    # ax2.grid(True, alpha=0.3, linestyle='--')
    # ax2.legend()
    
    # # Add overall title
    # fig.suptitle('Figure 1A: Time Series of Income and Growth Rates\nAlbuquerque, NM ACS Data', 
    #              fontsize=18, fontweight='bold', y=0.95)
    
    # # Add statistics text
    # stats_text = f'Block Groups: {n_block_groups}\nYears: {n_timesteps}\nTotal observations: {n_block_groups * n_timesteps}'
    # fig.text(0.02, 0.02, stats_text, fontsize=10,
    #          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    figure_path = os.path.join(save_path, 'figure1A_albuquerque_time_series.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 1A saved to: {figure_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(save_path, 'figure1A_albuquerque_time_series.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Figure 1A PDF saved to: {pdf_path}")
    
    return fig

def main():
    """Main function to create Figure 1A for Albuquerque, NM real data."""
    print("=" * 60)
    print("CREATING FIGURE 1A: ALBUQUERQUE, NM INCOME AND GROWTH RATE TIME SERIES")
    print("=" * 60)
    
    # Path to CBSA ACS data
    data_path = "figure1/cbsa_acs_data.pkl"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        print("Please ensure cbsa_acs_data.pkl is in the current directory.")
        return
    
    # Load Albuquerque data
    albuquerque_data = load_albuquerque_data(data_path)
    if albuquerque_data is None:
        return
    
    # Create Figure 1A
    print("\nCreating Figure 1A for Albuquerque, NM...")
    fig = create_figure1a(albuquerque_data)
    
    print("\n" + "=" * 60)
    print("FIGURE 1A CREATION COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - figure1/figure1A_albuquerque_time_series.png")
    print("  - figure1/figure1A_albuquerque_time_series.pdf")
    print("\nThe figure shows time series of:")
    print("  - Top: Log income trajectories for all block groups")
    print("  - Bottom: Growth rate trajectories for all block groups")
    print("  - Mean trajectories and confidence intervals")
    print("  - Colors indicate cumulative growth rate performance")

if __name__ == "__main__":
    main() 