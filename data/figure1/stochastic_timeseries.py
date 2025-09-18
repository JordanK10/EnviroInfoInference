#!/usr/bin/env python3
"""
Stochastic Time Series Generator
================================

This script generates a stochastic time series with low temporal correlations,
where y-values fluctuate between ln(2) and ln(0.5), and plots it on a log scale.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
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
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'ytick.labelsize': 16,
    'xtick.labelsize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def generate_stochastic_series(n_timesteps=1000, target_bounds=(np.log(0.5), np.log(2.0)), 
                              correlation_strength=0.5, noise_std=0.2, seed=42):
    """
    Generate a stochastic time series with low temporal correlations.
    
    Parameters:
    -----------
    n_timesteps : int
        Number of time steps to generate
    target_bounds : tuple
        (min_value, max_value) for the y-values in log space
    correlation_strength : float
        Strength of temporal correlation (0 = no correlation, 1 = perfect correlation)
    noise_std : float
        Standard deviation of the noise component
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (time_steps, y_values, parameters_used)
    """
    np.random.seed(seed)
    
    # Extract bounds
    y_min, y_max = target_bounds
    y_range = y_max - y_min
    y_center = (y_max + y_min) / 2
    
    print(f"Generating stochastic time series...")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Y bounds: [{y_min:.4f}, {y_max:.4f}] (log space)")
    print(f"  Y range: {y_range:.4f}")
    print(f"  Correlation strength: {correlation_strength}")
    print(f"  Noise std: {noise_std}")
    
    # Generate base random walk with low correlation
    if correlation_strength > 0:
        # AR(1) process: y_t = ρ*y_{t-1} + ε_t
        rho = correlation_strength
        y_values = np.zeros(n_timesteps)
        y_values[0] = y_center  # Start at center
        
        for t in range(1, n_timesteps):
            # Add correlation to previous value
            correlated_component = rho * y_values[t-1]
            # Add random noise
            noise = np.random.normal(0, noise_std)
            # Combine and ensure bounds
            y_values[t] = int((correlated_component + noise)*10)/2.
    else:
        # Pure random walk (no correlation)
        y_values = int(np.random.normal(y_center, noise_std, n_timesteps)*10)/2.
    
    
    # Create time array
    time_steps = np.arange(n_timesteps)
    
    # Calculate actual statistics
    actual_min = y_values.min()
    actual_max = y_values.max()
    actual_std = y_values.std()
    
    # Calculate temporal correlation (lag-1 autocorrelation)
    if len(y_values) > 1:
        lag1_correlation = np.corrcoef(y_values[:-1], y_values[1:])[0, 1]
    else:
        lag1_correlation = 0
    
    print(f"  Actual Y range: [{actual_min:.4f}, {actual_max:.4f}]")
    print(f"  Actual std: {actual_std:.4f}")
    print(f"  Lag-1 autocorrelation: {lag1_correlation:.4f}")
    
    parameters_used = {
        'n_timesteps': n_timesteps,
        'target_bounds': target_bounds,
        'correlation_strength': correlation_strength,
        'noise_std': noise_std,
        'seed': seed,
        'actual_bounds': (actual_min, actual_max),
        'actual_std': actual_std,
        'lag1_correlation': lag1_correlation
    }
    
    return time_steps, y_values, parameters_used

def plot_stochastic_series(time_steps, y_values, parameters, save_path="figure1"):
    """
    Plot the stochastic time series on a log scale.
    
    Parameters:
    -----------
    time_steps : np.ndarray
        Time steps array
    y_values : np.ndarray
        Y values array (in log space)
    parameters : dict
        Dictionary of parameters used
    save_path : str
        Directory to save the figure
    """
    # Create figure with two subplots
    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 4))
    y_values = np.array([1,-1,-1,1,-1,1,-1,-1,1,-1,1])
    # y_values = np.array([1,1,1,-1,-1,-1,1,1,-1,-1,-1])
    
    # Plot 1: Time series on log scale
    ax1.plot( y_values, color='#172741', linewidth=4, alpha=0.8)
    
    # Set x-axis ticks manually
    ax1.set_xticks([0, 5, 10])
    ax1.set_xticklabels(['0', '5', '10'])
    
    # Add horizontal lines for bounds
    y_min, y_max = parameters['target_bounds']

    ax1.axhline(y=(y_min + y_max)/2, color='#483326', linestyle=':', alpha=0.7, 
                label=f'Center: {(y_min + y_max)/2:.4f}',linewidth=3)
    
    ax1.set_ylim(-1.1, 1.1)
   
    # Adjust layout
    plt.tight_layout()
    
    # Also save as PDF for publication
    pdf_path = os.path.join(save_path, 'stochastic_timeseries.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Figure PDF saved to: {pdf_path}")
    
    return fig

def main():
    """Main function to generate and plot stochastic time series."""
    print("=" * 60)
    print("STOCHASTIC TIME SERIES GENERATOR")
    print("=" * 60)
    
    # Parameters for the stochastic series
    n_timesteps = 10
    target_bounds = (-1.0, 1.0)  # -1 to 1
    correlation_strength = 0.1  # Low temporal correlation
    noise_std = 0.1
    seed = 42
    
    print(f"Target bounds: [{target_bounds[0]:.4f}, {target_bounds[1]:.4f}]")
    print(f"Values will fluctuate between {target_bounds[0]:.1f} and {target_bounds[1]:.1f}")
    
    # Generate the stochastic time series
    time_steps, y_values, parameters = generate_stochastic_series(
        n_timesteps=n_timesteps,
        target_bounds=target_bounds,
        correlation_strength=correlation_strength,
        noise_std=noise_std,
        seed=seed
    )
    
    # Plot the results
    print("\nCreating plots...")
    fig = plot_stochastic_series(time_steps, y_values, parameters)
    
    print("\n" + "=" * 60)
    print("STOCHASTIC TIME SERIES GENERATION COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - figure1/stochastic_timeseries.png")
    print("  - figure1/stochastic_timeseries.pdf")
    print("\nThe figure shows:")
    print("  - Time series with low temporal correlation")
    print("  - Y-values fluctuate between -1.0 and 1.0")
    print("  - Simple single-panel plot")

if __name__ == "__main__":
    main() 