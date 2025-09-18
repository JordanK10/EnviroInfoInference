#!/usr/bin/env python3
"""
Variational Inference for Environmental Parameters (p and l)
==========================================================

This script implements a two-stage inference process:
1. Calculate p_t series (environmental predictability) from growth rate data
2. Use p_t series to infer l_t values (environmental payouts) using SSM model

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import pandas as pd
import warnings
import os
import sys
from pathlib import Path

# Add current directory to path for imports


from inference.ssm_model import fit_l_cross_sectional, estimate_l_time_series_bayesian

def frequentist_p_estimation(growth_rates_data, population_weights=None, 
                           min_agents_per_timestep=1):
    """
    Calculate p_t series from growth rate data using population-weighted frequentist approach.
    
    This is the improved version from urbandata that handles:
    - Population weighting
    - Missing data handling
    - Proper clipping for logit transforms
    - Validation of minimum data requirements
    
    Args:
        growth_rates_data: List of arrays, where each array contains growth rates for one agent
                          across all timesteps. Shape: (n_agents, n_timesteps)
        population_weights: Optional list of population weights for each agent. 
                          If None, equal weights are used.
        min_agents_per_timestep: Minimum number of agents required per timestep for valid p_t
    
    Returns:
        tuple: (p_t_series, timesteps) where p_t_series is array of p values for each timestep
    """
    
    if not growth_rates_data:
        raise ValueError("growth_rates_data cannot be empty")
    
    n_agents = len(growth_rates_data)
    n_timesteps = len(growth_rates_data[0]) if growth_rates_data else 0
    
    if n_timesteps == 0:
        raise ValueError("No timesteps found in growth rate data")
    
    # Set default population weights if not provided
    if population_weights is None:
        population_weights = [1.0] * n_agents
    elif len(population_weights) != n_agents:
        raise ValueError(f"population_weights length ({len(population_weights)}) must match "
                        f"number of agents ({n_agents})")
    
    # Initialize storage for each timestep
    timestep_wins = {}
    timestep_totals = {}
    
    print(f"Calculating p_t series for {n_agents} agents across {n_timesteps} timesteps...")
    
    # Process each timestep
    for timestep in range(n_timesteps):
        timestep_wins[timestep] = 0.0
        timestep_totals[timestep] = 0.0
        
        # Collect data for this timestep across all agents
        for agent_idx, agent_growth_rates in enumerate(growth_rates_data):
            if timestep < len(agent_growth_rates):
                y = agent_growth_rates[timestep]
                weight = population_weights[agent_idx]
                
                # Only count valid observations
                if np.isfinite(y) and weight > 0:
                    timestep_totals[timestep] += weight
                    if y > 0:  # Win condition
                        timestep_wins[timestep] += weight
    
    # Filter timesteps with sufficient data
    valid_timesteps = []
    p_t_series = []
    
    for timestep in range(n_timesteps):
        if timestep_totals[timestep] >= min_agents_per_timestep:
            p_t = timestep_wins[timestep] / timestep_totals[timestep]
            # Ensure p_t is within (0, 1) for logit transform
            p_t = np.clip(p_t, 1e-5, 1 - 1e-5)
            
            valid_timesteps.append(timestep)
            p_t_series.append(p_t)
            
            print(f"  Timestep {timestep}: p_t = {p_t:.3f} "
                  f"(wins: {timestep_wins[timestep]:.1f}, "
                  f"total: {timestep_totals[timestep]:.1f})")
    
    if not valid_timesteps:
        raise ValueError(f"No timesteps have sufficient data (minimum {min_agents_per_timestep} agents)")
    
    p_t_series = np.array(p_t_series)
    print(f"\np_t series calculated: {len(p_t_series)} timesteps")
    print(f"p_t range: {p_t_series.min():.3f} to {p_t_series.max():.3f}")
    print(f"Mean p_t: {np.mean(p_t_series):.3f}")
    
    return p_t_series, valid_timesteps

def bayesian_p_estimation(growth_rates_data, population_weights=None, 
                         growth_rate_uncertainties=None,
                         min_agents_per_timestep=1, prior_alpha=1, prior_beta=1,
                         verbose=True):
    """
    Calculate p_t series from growth rate data using Bayesian Binomial model.
    
    This approach models the number of "wins" (positive growth rates) at each timestep
    using a Binomial likelihood with a Beta prior, yielding full posterior distributions
    for p_t instead of just point estimates.
    
    NEW: Handles uncertainty in growth rates using soft classification. Instead of
    hard binary wins/losses, calculates the probability that each agent's true
    growth rate is positive, accounting for measurement uncertainty.
    
    Args:
        growth_rates_data: List of arrays, where each array contains growth rates for one agent
                          across all timesteps. Shape: (n_agents, n_timesteps)
        population_weights: Optional list of population weights for each agent. 
                          If None, equal weights are used.
        growth_rate_uncertainties: Optional list of arrays with standard errors of growth rates.
                                 If None, assumes no uncertainty (hard binary classification).
                                 Shape: (n_agents, n_timesteps)
        min_agents_per_timestep: Minimum number of agents required per timestep for valid p_t
        prior_alpha: Alpha parameter for Beta prior (default: 1 for uniform prior)
        prior_beta: Beta parameter for Beta prior (default: 1 for uniform prior)
    
    Returns:
        tuple: (p_t_series, p_t_posteriors, timesteps) where:
               - p_t_series: Array of posterior mean p values for each timestep
               - p_t_posteriors: List of Beta distribution objects for each timestep
               - timesteps: List of valid timestep indices
    """
    
    if not growth_rates_data:
        raise ValueError("growth_rates_data cannot be empty")
    
    n_agents = len(growth_rates_data)
    n_timesteps = len(growth_rates_data[0]) if growth_rates_data else 0
    
    if n_timesteps == 0:
        raise ValueError("No timesteps found in growth rate data")
    
    # Set default population weights if not provided
    if population_weights is None:
        population_weights = [1.0] * n_agents
    elif len(population_weights) != n_agents:
        raise ValueError(f"population_weights length ({len(population_weights)}) must match "
                        f"number of agents ({n_agents})")
    
    # Set default uncertainties if not provided
    if growth_rate_uncertainties is None:
        growth_rate_uncertainties = [np.zeros_like(agent_growth_rates) for agent_growth_rates in growth_rates_data]
        if verbose:
            print("  No growth rate uncertainties provided - using hard binary classification")
    elif len(growth_rate_uncertainties) != n_agents:
        raise ValueError(f"growth_rate_uncertainties length ({len(growth_rate_uncertainties)}) must match "
                        f"number of agents ({n_agents})")
    
    # Initialize storage for each timestep
    timestep_expected_wins = {}
    timestep_totals = {}
    
    if verbose:
        print(f"Calculating Bayesian p_t series for {n_agents} agents across {n_timesteps} timesteps...")
        # This condition is tricky as we default uncertainties to zeros if None.
        # A more robust check is if any uncertainty value is actually greater than zero.
        has_uncertainty = any(np.any(u > 0) for u in growth_rate_uncertainties)
        if has_uncertainty:
            print("  Using soft classification with growth rate uncertainties")
    
    # Process each timestep
    for timestep in range(n_timesteps):
        timestep_expected_wins[timestep] = 0.0
        timestep_totals[timestep] = 0.0
        
        # Collect data for this timestep across all agents
        for agent_idx, agent_growth_rates in enumerate(growth_rates_data):
            if timestep < len(agent_growth_rates):
                y = agent_growth_rates[timestep]
                y_std = growth_rate_uncertainties[agent_idx][timestep] if timestep < len(growth_rate_uncertainties[agent_idx]) else 0.0
                weight = population_weights[agent_idx]
                
                # Only count valid observations
                if np.isfinite(y) and weight > 0:
                    timestep_totals[timestep] += weight
                    
                    # Calculate probability of win accounting for uncertainty
                    if y_std > 0:
                        # Soft classification: calculate P(y_true > 0 | y_observed, y_std)
                        from scipy.stats import norm
                        prob_win = 1 - norm.cdf(0, loc=y, scale=y_std)
                        # Clip to valid probability range
                        prob_win = np.clip(prob_win, 0.0, 1.0)
                        timestep_expected_wins[timestep] += prob_win * weight
                    else:
                        # Hard classification: no uncertainty
                        if y > 0:
                            timestep_expected_wins[timestep] += weight
    
    # Filter timesteps with sufficient data and calculate Bayesian posteriors
    valid_timesteps = []
    p_t_series = []
    p_t_posteriors = []
    
    from scipy.stats import beta
    
    for timestep in range(n_timesteps):
        if timestep_totals[timestep] >= min_agents_per_timestep:
            expected_wins = timestep_expected_wins[timestep]
            total = timestep_totals[timestep]
            
            # Calculate posterior parameters for Beta distribution
            # Posterior: Beta(alpha' = prior_alpha + expected_wins, beta' = prior_beta + total - expected_wins)
            posterior_alpha = prior_alpha + expected_wins
            posterior_beta = prior_beta + total - expected_wins
            
            # Create Beta distribution object for this timestep
            posterior_dist = beta(posterior_alpha, posterior_beta)
            
            # Calculate posterior mean (this is our point estimate)
            p_t_mean = posterior_dist.mean()
            
            # Ensure p_t is within (0, 1) for logit transform
            p_t_mean = np.clip(p_t_mean, 1e-5, 1 - 1e-5)
            
            valid_timesteps.append(timestep)
            p_t_series.append(p_t_mean)
            p_t_posteriors.append(posterior_dist)
            
            if verbose:
                # Calculate credible interval (95%)
                ci_lower, ci_upper = posterior_dist.interval(0.95)
                
                # Show classification details
                has_uncertainty = any(np.any(u > 0) for u in growth_rate_uncertainties)
                if has_uncertainty:
                    print(f"  Timestep {timestep}: p_t = {p_t_mean:.3f} "
                          f"(95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]) "
                          f"(expected wins: {expected_wins:.2f}, total: {total:.1f})")
                else:
                    print(f"  Timestep {timestep}: p_t = {p_t_mean:.3f} "
                          f"(95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]) "
                          f"(wins: {expected_wins:.1f}, total: {total:.1f})")
    
    if not valid_timesteps:
        raise ValueError(f"No timesteps have sufficient data (minimum {min_agents_per_timestep} agents)")
    
    p_t_series = np.array(p_t_series)
    if verbose:
        print(f"\n✓ Bayesian p_t series calculated: {len(p_t_series)} timesteps")
        print(f"p_t range: {p_t_series.min():.3f} to {p_t_series.max():.3f}")
        print(f"Mean p_t: {np.mean(p_t_series):.3f}")
    
    return p_t_series, p_t_posteriors, valid_timesteps

def infer_l_t_series(growth_rates_data, p_t_series, population_weights=None, 
                    delta=0.05, n_samples=8000, rolling_window=3):
    """
    Infer l_t series using the cross-sectional Bayesian approach from SSM model.
    
    Args:
        growth_rates_data: List of arrays, where each array contains growth rates for one agent
                          across all timesteps. Shape: (n_agents, n_timesteps)
        p_t_series: Array of p_t values for each timestep
        population_weights: Optional population weights for each agent
        delta: Sub-optimality offset for agent beliefs
        n_samples: Number of VI samples per timestep
        rolling_window: Window size for temporal smoothing
    
    Returns:
        dict: Dictionary mapping timestep -> smoothed l estimate
    """
    
    if not growth_rates_data:
        raise ValueError("growth_rates_data cannot be empty")
    
    n_agents = len(growth_rates_data)
    n_timesteps = len(p_t_series)
    
    # Set default population weights if not provided
    if population_weights is None:
        population_weights = [1.0] * n_agents
    
    print(f"\nInferring l_t series for {n_timesteps} timesteps...")
    
    # Step 1: Collect all growth rates by timestep
    growth_rates_by_timestep = {}
    population_by_timestep = {}
    
    for agent_idx, agent_growth_rates in enumerate(growth_rates_data):
        agent_weight = population_weights[agent_idx]
        
        for timestep, y in enumerate(agent_growth_rates):
            if timestep < n_timesteps and np.isfinite(y):
                if timestep not in growth_rates_by_timestep:
                    growth_rates_by_timestep[timestep] = []
                    population_by_timestep[timestep] = []
                growth_rates_by_timestep[timestep].append(y)
                population_by_timestep[timestep].append(agent_weight)
    
    # Step 2: Run cross-sectional VI for each timestep
    l_estimates = {}
    
    for timestep in range(n_timesteps):
        if timestep in growth_rates_by_timestep:
            y_values_t = growth_rates_by_timestep[timestep]
            p_hat_t = p_t_series[timestep]
            pop_weights_t = population_by_timestep[timestep]
            
            try:
                # Run the cross-sectional Bayesian model with weights
                idata = fit_l_cross_sectional(y_values_t, p_hat_t, pop_weights_t, delta, n_samples)
                
                # Extract the posterior mean as our point estimate
                l_posterior_samples = idata.posterior["l"].values.flatten()
                l_est_t = np.mean(l_posterior_samples)
                l_estimates[timestep] = l_est_t
                
                print(f"  Timestep {timestep}: l_est = {l_est_t:.3f} "
                      f"(from {len(y_values_t)} agents)")
                
            except Exception as e:
                print(f"  Timestep {timestep}: Failed to estimate l ({str(e)})")
                # Use a reasonable default if estimation fails
                l_estimates[timestep] = 2.0
    
    # Step 3: Apply temporal smoothing (rolling mean)
    if len(l_estimates) >= rolling_window:
        print(f"\nApplying temporal smoothing (rolling window = {rolling_window})...")
        
        # Convert to sorted lists for rolling mean calculation
        sorted_timesteps = sorted(l_estimates.keys())
        l_values = [l_estimates[t] for t in sorted_timesteps]
        
        # Apply rolling mean
        l_smoothed_values = []
        for i in range(len(l_values)):
            # Define the window around the current point
            start_idx = max(0, i - rolling_window // 2)
            end_idx = min(len(l_values), i + rolling_window // 2 + 1)
            window_values = l_values[start_idx:end_idx]
            l_smoothed = np.mean(window_values)
            l_smoothed_values.append(l_smoothed)
        
        # Create the final smoothed dictionary
        l_smoothed_dict = {}
        for i, timestep in enumerate(sorted_timesteps):
            l_smoothed_dict[timestep] = l_smoothed_values[i]
            print(f"  Timestep {timestep}: l_raw = {l_estimates[timestep]:.3f} -> "
                  f"l_smooth = {l_smoothed_dict[timestep]:.3f}")
        
        return l_smoothed_dict
    
    else:
        print(f"Not enough timesteps for smoothing ({len(l_estimates)} < {rolling_window})")
        return l_estimates

