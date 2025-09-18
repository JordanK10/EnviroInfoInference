#!/usr/bin/env python3
"""
Figure 1A: Log Incomes vs Growth Rates
======================================

This script creates Figure 1A for the paper, showing the relationship between
log incomes and growth rates from the synthetic data.

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
        Array of shape (n_agents, n_timesteps + 1) with resource values
        
    Returns:
    --------
    np.ndarray
        Array of shape (n_agents, n_timesteps) with growth rates
    """
    # Calculate growth rates: (r_t - r_{t-1}) / r_{t-1}
    growth_rates = np.diff(resources, axis=1) / resources[:, :-1]
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
    
    # Sort agents by cumulative growth rate for consistent coloring
    sorted_indices = np.argsort(cumulative_growth_rates)
    
    # Group block groups into three equal-sized quantiles
    n_per_group = n_block_groups // 3
    bottom_third = sorted_indices[:n_per_group]
    middle_third = sorted_indices[n_per_group:2*n_per_group]
    top_third = sorted_indices[2*n_per_group:]
    
    print(f"Grouped block groups: Bottom third (n={len(bottom_third)}), Middle third (n={len(middle_third)}), Top third (n={len(top_third)})")
    
    # Create the figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
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
        
    # Custom color spectrum for quantiles
    # custom_colors1 = ['#0F0F11', '#AD8F78', '#C7D4E4']
    
    custom_colors1 = ['#172741', '#5F8089', '#C2C4B8', '#C3A983', '#483326']   
    # Plot 1: Time series of log incomes for all agents
    print("Plotting log income time series...")
    print(f"Cumulative growth rates range: {cumulative_growth_rates.min():.3f} to {cumulative_growth_rates.max():.3f}")
    print(f"Using relative growth ranks (percentiles) for coloring")
    
    # Create color mapping based on relative growth ranks
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', custom_colors1)
    
    # Normalize relative growth ranks for color mapping (0 to 1)
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=1)
    
    # Calculate empirical confidence intervals based on actual data
    print("Calculating empirical confidence intervals...")
    
    # Calculate empirical mean and standard deviation at each timestep
    empirical_mean = np.mean(log_incomes, axis=0)
    empirical_std = np.std(log_incomes, axis=0)
    
    # Calculate 95% CI bounds around the empirical mean trajectory
    ci_lower = empirical_mean - 1.96 * empirical_std 
    ci_upper = empirical_mean + 1.96 * empirical_std 
    
    # Plot the empirical confidence interval
    ax1.fill_between(time_steps, ci_lower, ci_upper, 
                     color='#3D3D3D', alpha=0.5, label='Empirical 95% CI (Data)',zorder=100)
    
    # Plot the empirical mean trajectory
    ax1.plot(time_steps, empirical_mean, color='#3D3D3D', linewidth=2, 
             linestyle='--', alpha=0.8, label='Empirical Mean Trajectory')
    
    # Reassign quantiles based on final timestep position relative to empirical CI
    print("Reassigning quantiles based on empirical CI...")
    
    final_log_incomes = log_incomes[:, -1]  # Final timestep values
    final_ci_lower = ci_lower[-1]
    final_ci_upper = ci_upper[-1]
    
    print(f"Final timestep CI: [{final_ci_lower:.3f}, {final_ci_upper:.3f}]")
    print(f"Final log incomes range: [{final_log_incomes.min():.3f}, {final_log_incomes.max():.3f}]")
    
    # Remove quantile logic - we'll just plot all trajectories in order
    print("Plotting all trajectories colored by growth rate order...")
    
    # Calculate 95% confidence intervals for each quantile group
    print("Calculating 95% confidence intervals for quantile groups...")
    
    # Bottom quantile CI
    if len(bottom_agents) > 1:
        bottom_std = np.std(bottom_incomes, axis=0)
        bottom_ci = 1.96 * bottom_std 
        bottom_ci_lower = bottom_mean - bottom_ci
        bottom_ci_upper = bottom_mean + bottom_ci
        print(f"Bottom quantile CI calculated (n={len(bottom_agents)})")
    else:
        bottom_ci_lower = bottom_mean
        bottom_ci_upper = bottom_mean
        print(f"Bottom quantile has only {len(bottom_agents)} agent(s), no CI")
    
    # Middle quantile CI
    if len(middle_agents) > 1:
        middle_std = np.std(middle_incomes, axis=0)
        middle_ci = 1.96 * middle_std 
        middle_ci_lower = middle_mean - middle_ci
        middle_ci_upper = middle_mean + middle_ci
        print(f"Middle quantile CI calculated (n={len(middle_agents)})")
    else:
        middle_ci_lower = middle_mean
        middle_ci_upper = middle_mean
        print(f"Middle quantile has only {len(middle_agents)} agent(s), no CI")
    
    # Top quantile CI
    if len(top_agents) > 1:
        top_std = np.std(top_incomes, axis=0)
        top_ci = 1.96 * top_std 
        top_ci_lower = top_mean - top_ci
        top_ci_upper = top_mean + top_ci
        print(f"Top quantile CI calculated (n={len(top_agents)})")
    else:
        top_ci_lower = top_mean
        top_ci_upper = top_mean
        print(f"Top quantile has only {len(top_agents)} agent(s), no CI")
    

    
    
    # Top quantile CI
    if len(top_agents) > 1:
        top_std = np.std(top_incomes, axis=0)
        top_ci = 1.96 * top_std 
        top_ci_lower = top_mean - top_ci
        top_ci_upper = top_mean + top_ci
        print(f"Top quantile CI calculated (n={len(top_agents)})")
    else:
        top_ci_lower = top_mean
        top_ci_upper = top_mean
        print(f"Top quantile has only {len(top_agents)} agent(s), no CI")
    

    
    # Plot 95% confidence intervals for each quantile group
    print("Plotting quantile confidence intervals...")
    if len(bottom_agents) > 1:
        ax1.fill_between(time_steps, bottom_ci_lower, bottom_ci_upper, 
                         color=quantile_colors[0], alpha=0.2, label='Bottom Quantile 95% CI')
    
    if len(middle_agents) > 1:
        ax1.fill_between(time_steps, middle_ci_lower, middle_ci_upper, 
                         color=quantile_colors[1], alpha=0.2, label='Middle Quantile 95% CI')
    
    if len(top_agents) > 1:
        ax1.fill_between(time_steps, top_ci_lower, top_ci_upper, 
                         color=quantile_colors[2], alpha=0.2, label='Top Quantile 95% CI')
    
    # Plot individual trajectories with reduced alpha
    print("Plotting individual trajectories...")
    
    # Bottom quantile trajectories
    for block_group_id in bottom_agents:
        ax1.plot(time_steps, log_incomes[block_group_id, :], 
                color='#3d3d3d', alpha=0.2, linewidth=0.8,zorder=-1)
    
    # Middle quantile trajectories
    for block_group_id in middle_agents:
        ax1.plot(time_steps, log_incomes[block_group_id, :], 
                color='#3d3d3d', alpha=0.2, linewidth=0.8,zorder=-1)
    
    # Top quantile trajectories
    for block_group_id in top_agents:
        ax1.plot(time_steps, log_incomes[block_group_id, :], 
                color='#3d3d3d', alpha=0.2, linewidth=0.8,zorder=-1)

    # Plot mean trajectories for each quantile group
    ax1.plot(time_steps, bottom_mean, color=quantile_colors[0], linewidth=3, alpha=0.8, 
             label=f'Bottom Quantile Mean (n={len(bottom_agents)})')
    
    ax1.plot(time_steps, middle_mean, color=quantile_colors[1], linewidth=3, alpha=0.8, 
             label=f'Middle Quantile Mean (n={len(middle_agents)})')
    
    ax1.plot(time_steps, top_mean, color=quantile_colors[2], linewidth=3, alpha=0.8, 
             label=f'Top Quantile Mean (n={len(top_agents)})')
    
    # ax1.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    # ax1.set_ylabel('Log Income (log dollars)', fontsize=14, fontweight='bold')
    # ax1.set_title('Time Series of Log Incomes\n(Variance-Informed Quantiles)', fontsize=16, fontweight='bold')
    # ax1.grid(True, alpha=0.3, linestyle='--')
    # ax1.legend()
    
    # Plot 2: Time series of growth rates for all agents
    print("Plotting growth rate time series...")
    
    # Plot growth rates colored by quantile membership
    for block_group_id in bottom_agents:
        ax2.plot(time_steps_growth, growth_rates[block_group_id, :], 
                color=quantile_colors[0], alpha=0.2, linewidth=0.8)
    
    for block_group_id in middle_agents:
        ax2.plot(time_steps_growth, growth_rates[block_group_id, :], 
                color=quantile_colors[1], alpha=0.2, linewidth=0.8)
    
    for block_group_id in top_agents:
        ax2.plot(time_steps_growth, growth_rates[block_group_id, :], 
                color=quantile_colors[2], alpha=0.2, linewidth=0.8)
    
    # Plot mean growth rate trajectories for each group
    bottom_growth_mean = np.mean(growth_rates[bottom_agents, :], axis=0) if len(bottom_agents) > 0 else np.zeros(n_timesteps)
    middle_growth_mean = np.mean(growth_rates[middle_agents, :], axis=0) if len(middle_agents) > 0 else np.zeros(n_timesteps)
    top_growth_mean = np.mean(growth_rates[top_agents, :], axis=0) if len(top_agents) > 0 else np.zeros(n_timesteps)
    
    ax2.plot(time_steps_growth, bottom_growth_mean, color='white', linewidth=4, alpha=0.8, 
             label=f'Bottom Quantile Mean (n={len(bottom_agents)})')
    ax2.plot(time_steps_growth, middle_growth_mean, color='white', linewidth=4, alpha=0.8, 
             label=f'Middle Quantile Mean (n={len(middle_agents)})')
    ax2.plot(time_steps_growth, top_growth_mean, color='white', linewidth=4, alpha=0.8, 
             label=f'Top Quantile Mean (n={len(top_agents)})')

    ax2.plot(time_steps_growth, bottom_growth_mean, color=quantile_colors[0], linewidth=3, alpha=0.8, 
             label=f'Bottom Quantile Mean (n={len(bottom_agents)})')
    ax2.plot(time_steps_growth, middle_growth_mean, color=quantile_colors[1], linewidth=3, alpha=0.8, 
             label=f'Middle Quantile Mean (n={len(middle_agents)})')
    ax2.plot(time_steps_growth, top_growth_mean, color=quantile_colors[2], linewidth=3, alpha=0.8, 
             label=f'Top Quantile Mean (n={len(top_agents)})')
    

    
    # ax2.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    # ax2.set_ylabel('Growth Rate (fractional change)', fontsize=14, fontweight='bold')
    # ax2.set_title('Time Series of Growth Rates\n(Colored by Quantile Membership)', fontsize=16, fontweight='bold')
    # ax2.grid(True, alpha=0.3, linestyle='--')
    # ax2.legend()
    
    # Add colorbar to show quantile mapping
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import ScalarMappable
    # quantile_cmap = ListedColormap(quantile_colors)
    # sm = ScalarMappable(cmap=quantile_cmap, norm=Normalize(vmin=0, vmax=2))
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=[ax1, ax2], shrink=0.8, aspect=30, ticks=[0.33, 1, 1.67])
    # cbar.set_ticklabels(['Bottom Quantile', 'Middle Quantile', 'Top Quantile'])
    # cbar.set_label('Variance-Informed Quantiles', fontsize=12, fontweight='bold')
    
    # Add overall title
    # fig.suptitle('Figure 1A: Time Series of Income and Growth Rates\nSynthetic Agent Data', 
    #              fontsize=18, fontweight='bold', y=0.95)
    
    # Add statistics text
    # stats_text = f'Block Groups: {n_block_groups}\nTimesteps: {n_timesteps}\nTotal observations: {n_block_groups * n_timesteps}'
    # fig.text(0.02, 0.02, stats_text, fontsize=10,
    #          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    figure_path = os.path.join(save_path, 'figure1A_time_series.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure 1A saved to: {figure_path}")
    
    # Also save as PDF for publication
    pdf_path = os.path.join(save_path, 'figure1A_time_series.pdf')
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
    print("  - figure1/figure1A_time_series.png")
    print("  - figure1/figure1A_time_series.pdf")
    print("\nThe figure shows time series of:")
    print("  - Top: Log income trajectories for all block groups")
    print("  - Bottom: Growth rate trajectories for all block groups")
    print("  - Mean and confidence intervals for each quantile")

if __name__ == "__main__":
    main() 