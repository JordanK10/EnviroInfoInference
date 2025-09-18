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

def load_dummy_data(data_path):
    """
    Load the dummy data from the pickle file.
    
    Parameters:
    -----------
    data_path : str
        Path to the dummy data pickle file
        
    Returns:
    --------
    dict
        Loaded dummy data
    """
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded dummy data from {data_path}")
        return data
    except Exception as e:
        print(f"✗ Error loading dummy data: {e}")
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

def create_figure1a(dummy_data, save_path="figure1"):
    """
    Create Figure 1A: Time Series of Income vs Growth Rates.
    
    Parameters:
    -----------
    dummy_data : dict
        Loaded dummy data
    save_path : str
        Directory to save the figure
    """
    # Extract data
    resources = dummy_data['agent_trajectories']['resources']  # Shape: (n_agents, n_timesteps + 1)
    n_agents, n_timesteps_plus_1 = resources.shape
    n_timesteps = n_timesteps_plus_1 - 1
    
    print(f"Data shape: {n_agents} agents, {n_timesteps} timesteps")
    
    # Calculate growth rates
    growth_rates = calculate_growth_rates(resources)  # Shape: (n_agents, n_timesteps)
    
    # Calculate log incomes (using resources at each timestep)
    log_incomes = np.log(resources) # Shape: (n_agents, n_timesteps + 1)
    
    # Calculate cumulative growth rates for coloring
    cumulative_growth_rates = np.zeros(n_agents)
    for agent_id in range(n_agents):
        # Calculate total growth from start to end
        initial_resources = resources[agent_id, 0]
        final_resources = resources[agent_id, -1]
        cumulative_growth_rates[agent_id] = (np.log(final_resources) - np.log(initial_resources)) 
    
    # Convert to relative growth rate (percentile rank) for coloring
    from scipy.stats import rankdata
    relative_growth_ranks = rankdata(cumulative_growth_rates) / len(cumulative_growth_rates)  # 0 to 1
    
    # Sort agents by cumulative growth rate for consistent coloring
    sorted_indices = np.argsort(cumulative_growth_rates)
    
    # Group agents into three equal-sized quantiles
    n_per_group = n_agents // 3
    bottom_third = sorted_indices[:n_per_group]
    middle_third = sorted_indices[n_per_group:2*n_per_group]
    top_third = sorted_indices[2*n_per_group:]
    
    print(f"Grouped agents: Bottom third (n={len(bottom_third)}), Middle third (n={len(middle_third)}), Top third (n={len(top_third)})")
    
    # Create the figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Create time array for x-axis
    time_steps = np.arange(n_timesteps + 1)
    time_steps_growth = np.arange(1, n_timesteps + 1)  # Growth rates start from timestep 1
    
    # Custom color spectrum for quantiles
    # custom_colors1 = ['#0F0F11', '#AD8F78', '#C7D4E4']
    
    custom_colors1 = ['#172741', '#5F8089', '#C2C4B8', '#C3A983', '#483326']   
    quantile_colors = ['#172741', '#C2C4B8', '#483326']   
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
    
    # Calculate variance-informed confidence intervals based on theoretical equation
    print("Calculating variance-informed confidence intervals...")
    
    # Parameters from your specification
    l = 2.0
    w = 10.0
    p = 0.7
    x = 0.6
    
    # Calculate theoretical growth rate per timestep
    theoretical_growth_rate = p * np.log(l * x) + (1 - p) * np.log((1 - x) * l)
    print(f"Theoretical growth rate per timestep: {theoretical_growth_rate:.6f}")
    
    # Calculate variance at each timestep: σ² = p(1-p)(ln((x(l-1)/(1-x)))²/sqrt(w)
    variance_factor = p * (1 - p) * (np.log((x * (l - 1)) / (1 - x))) ** 2 / np.sqrt(w)
    print(f"Variance factor: {variance_factor:.6f}")
    
    # Calculate theoretical trajectory starting from the initial mean
    initial_mean = np.mean(log_incomes[:, 0])
    theoretical_trajectory = initial_mean + theoretical_growth_rate * time_steps
    
    # Calculate theoretical standard deviation at each timestep
    # For log resources, variance accumulates over time
    theoretical_std = np.sqrt(variance_factor * time_steps)
    
    # Calculate 95% CI bounds around the theoretical trajectory
    ci_lower = theoretical_trajectory - 1.96 * theoretical_std
    ci_upper = theoretical_trajectory + 1.96 * theoretical_std
    
    # Plot the theoretical confidence interval
    ax1.fill_between(time_steps, ci_lower, ci_upper, 
                     color='#3D3D3D', alpha=0.5, label='Theoretical 95% CI (σ-informed)',zorder=100)
    
    # Plot the theoretical trajectory
    ax1.plot(time_steps, theoretical_trajectory, color='#3D3D3D', linewidth=2, 
             linestyle='--', alpha=0.8, label='Theoretical Trajectory')
    
    # Reassign quantiles based on final timestep position relative to CI
    print("Reassigning quantiles based on variance-informed CI...")
    
    final_log_incomes = log_incomes[:, -1]  # Final timestep values
    final_ci_lower = ci_lower[-1]
    final_ci_upper = ci_upper[-1]
    
    print(f"Final timestep CI: [{final_ci_lower:.3f}, {final_ci_upper:.3f}]")
    print(f"Final log incomes range: [{final_log_incomes.min():.3f}, {final_log_incomes.max():.3f}]")
    
    # Reassign agents to quantiles based on final position
    bottom_agents = np.where(final_log_incomes < final_ci_lower)[0]
    middle_agents = np.where((final_log_incomes >= final_ci_lower) & (final_log_incomes <= final_ci_upper))[0]
    top_agents = np.where(final_log_incomes > final_ci_upper)[0]
    
    print(f"Reassigned quantiles: Bottom (n={len(bottom_agents)}), Middle (n={len(middle_agents)}), Top (n={len(top_agents)})")
    
    # Recalculate group means for the new quantiles
    bottom_incomes = log_incomes[bottom_agents, :] if len(bottom_agents) > 0 else np.zeros((1, n_timesteps + 1))
    top_incomes = log_incomes[top_agents, :] if len(top_agents) > 0 else np.zeros((1, n_timesteps + 1))
    
    bottom_mean = np.mean(bottom_incomes, axis=0) if len(bottom_agents) > 0 else np.zeros(n_timesteps + 1)
    top_mean = np.mean(top_incomes, axis=0) if len(top_agents) > 0 else np.zeros(n_timesteps + 1)
    
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
    
    if len(top_agents) > 1:
        ax1.fill_between(time_steps, top_ci_lower, top_ci_upper, 
                         color=quantile_colors[2], alpha=0.2, label='Top Quantile 95% CI')
    
    # Plot individual trajectories with reduced alpha
    print("Plotting individual trajectories...")
    
    # Bottom quantile trajectories
    for agent_id in bottom_agents:
        ax1.plot(time_steps, log_incomes[agent_id, :], 
                color='#3d3d3d', alpha=0.05, linewidth=0.8,zorder=-1)
    
    # Middle quantile trajectories
    for agent_id in middle_agents:
        ax1.plot(time_steps, log_incomes[agent_id, :], 
                color='#3d3d3d', alpha=0.05, linewidth=0.8,zorder=-1)
    
    # Top quantile trajectories
    for agent_id in top_agents:
        ax1.plot(time_steps, log_incomes[agent_id, :], 
                color='#3d3d3d', alpha=0.05, linewidth=0.8,zorder=-1)
    
    # Plot mean trajectories for each group
    ax1.plot(time_steps, bottom_mean, color=quantile_colors[0], linewidth=3, alpha=0.8, 
             label=f'Bottom Quantile Mean (n={len(bottom_agents)})')

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
    for agent_id in bottom_agents:
        ax2.plot(time_steps_growth, growth_rates[agent_id, :], 
                color=quantile_colors[0], alpha=0.2, linewidth=0.8)
    
    for agent_id in middle_agents:
        ax2.plot(time_steps_growth, growth_rates[agent_id, :], 
                color=quantile_colors[1], alpha=0.2, linewidth=0.8)
    
    for agent_id in top_agents:
        ax2.plot(time_steps_growth, growth_rates[agent_id, :], 
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
    # stats_text = f'Agents: {n_agents}\nTimesteps: {n_timesteps}\nTotal observations: {n_agents * n_timesteps}'
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
    """Main function to create Figure 1A."""
    print("=" * 60)
    print("CREATING FIGURE 1A: INCOME AND GROWTH RATE TIME SERIES")
    print("=" * 60)
    
    # Path to dummy data
    data_path = "validation/validation_dynamic_p_l/dummy_data_kelly_betting_static_xSIMPLE.pkl"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        print("Please run generate_dummy_data.py first to create the data.")
        return
    
    # Load dummy data
    dummy_data = load_dummy_data(data_path)
    if dummy_data is None:
        return
    
    # Create Figure 1A
    print("\nCreating Figure 1A...")
    fig = create_figure1a(dummy_data)
    
    print("\n" + "=" * 60)
    print("FIGURE 1A CREATION COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - figure1/figure1A_time_series.png")
    print("  - figure1/figure1A_time_series.pdf")
    print("\nThe figure shows time series of:")
    print("  - Top: Log income trajectories for all agents")
    print("  - Bottom: Growth rate trajectories for all agents")
    print("  - Mean and median trajectories highlighted")

if __name__ == "__main__":
    main() 