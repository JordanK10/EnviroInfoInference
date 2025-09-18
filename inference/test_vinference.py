#!/usr/bin/env python3
"""
Test Script for Vinference
==========================

This script loads dummy data and tests the vinference functions with real data
instead of synthetic data.

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vinference import frequentist_p_estimation, bayesian_p_estimation, infer_l_t_series

def load_dummy_data(file_path):
    """
    Load dummy data from pickle file.
    
    Args:
        file_path: Path to the dummy data pickle file
        
    Returns:
        dict: Loaded dummy data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded dummy data from {file_path}")
        return data
    except Exception as e:
        print(f"✗ Error loading dummy data: {e}")
        return None

def extract_growth_rates_from_dummy_data(dummy_data):
    """
    Extract growth rate data from dummy data structure.
    
    Args:
        dummy_data: Loaded dummy data dictionary
        
    Returns:
        tuple: (growth_rates_data, population_weights, p_t_series)
    """
    print("\nExtracting data from dummy data structure...")
    
    # Print available keys to understand the structure
    print("Available keys in dummy data:")
    for key, value in dummy_data.items():
        if isinstance(value, (list, np.ndarray)):
            print(f"  {key}: {type(value)} with shape/length {len(value)}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Try to find growth rate data
    growth_rates_data = []
    population_weights = []
    
    # Look for different possible structures
    if 'vi_data' in dummy_data:
        print("\nFound 'vi_data' key - extracting growth rates...")
        vi_data = dummy_data['vi_data']
        
        for agent_data in vi_data:
            if 'income_growth_rates' in agent_data:
                growth_rates = agent_data['income_growth_rates']
                growth_rates_data.append(growth_rates)
                
                # Try to get population weight if available
                if 'population' in agent_data:
                    pop_weight = np.mean(agent_data['population'])
                else:
                    pop_weight = 1.0
                population_weights.append(pop_weight)
                
    
    elif 'agent_trajectories' in dummy_data:
        print("\nFound 'agent_trajectories' key - extracting growth rates...")
        trajectories = dummy_data['agent_trajectories']
        
        if 'resources' in trajectories:
            resources = trajectories['resources']
            n_agents, n_timesteps = resources.shape
            
            print(f"  Found {n_agents} agents with {n_timesteps} timesteps")
            
            # Calculate growth rates from resource trajectories
            for agent_idx in range(n_agents):
                agent_resources = resources[agent_idx, :]
                # Calculate growth rates: (r_t - r_{t-1}) / r_{t-1}
                growth_rates = np.diff(agent_resources) / agent_resources[:-1]
                growth_rates_data.append(growth_rates)
                population_weights.append(1.0)  # Default weight
    
    elif 'growth_rates' in dummy_data:
        print("\nFound 'growth_rates' key - extracting directly...")
        growth_rates_data = dummy_data['growth_rates']
        population_weights = [1.0] * len(growth_rates_data)
    
    else:
        # Look for any array that might be growth rate data
        for key, value in dummy_data.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                print(f"\nFound potential growth rate data in '{key}' key")
                print(f"  Shape: {value.shape}")
                
                # Assume first dimension is agents, second is timesteps
                n_agents, n_timesteps = value.shape
                
                for agent_idx in range(n_agents):
                    growth_rates_data.append(value[agent_idx, :])
                    population_weights.append(1.0)
                
                print(f"  Extracted {n_agents} agents with {n_timesteps} timesteps")
                break
    
    if not growth_rates_data:
        raise ValueError("Could not find growth rate data in the dummy data")
    
    # Get p_t_series if available
    p_t_series = None
    if 'p_t_series' in dummy_data:
        p_t_series = dummy_data['p_t_series']
        print(f"  Found p_t_series with {len(p_t_series)} timesteps")
    
    print(f"\n✓ Successfully extracted data:")
    print(f"  Growth rates: {len(growth_rates_data)} agents")
    print(f"  Population weights: {len(population_weights)} weights")
    print(f"  Timesteps per agent: {len(growth_rates_data[0]) if growth_rates_data else 0}")
    if p_t_series is not None:
        print(f"  p_t_series: {len(p_t_series)} timesteps")
    
    return growth_rates_data, population_weights, p_t_series

def test_vinference_with_dummy_data(dummy_data_file="dummy_data_kelly_betting_static_x.pkl"):
    """
    Test the vinference functions with dummy data.
    
    Args:
        dummy_data_file: Name of the dummy data file to load
    """
    print("=" * 60)
    print("TESTING VINFERENCE WITH DUMMY DATA")
    print("=" * 60)
    
    # Load dummy data
    print(f"\nLoading dummy data from {dummy_data_file}...")
    dummy_data = load_dummy_data(dummy_data_file)
    if dummy_data is None:
        return
    
    # Extract growth rate data
    try:
        growth_rates_data, population_weights, existing_p_t_series = extract_growth_rates_from_dummy_data(dummy_data)
    except Exception as e:
        print(f"✗ Failed to extract growth rate data: {e}")
        return
    
    try:
        # Stage 1: Calculate p_t series from growth rate data
        print("\n" + "=" * 40)
        print("STAGE 1: CALCULATING p_t SERIES")
        print("=" * 40)
        
        # Test both frequentist and Bayesian approaches
        print("\n--- FREQUENTIST APPROACH ---")
        p_t_frequentist, timesteps_freq = frequentist_p_estimation(
            growth_rates_data, 
            population_weights,
            min_agents_per_timestep=5  # Lower threshold for testing
        )
        
        print("\n--- BAYESIAN APPROACH ---")
        p_t_bayesian, p_t_posteriors, timesteps_bayes = bayesian_p_estimation(
            growth_rates_data, 
            population_weights,
            growth_rate_uncertainties=None,  # No uncertainties for now
            min_agents_per_timestep=5,  # Lower threshold for testing
            prior_alpha=1,  # Uniform prior
            prior_beta=1
        )
        
        # Use Bayesian results for the main inference (more robust)
        p_t_series = p_t_bayesian
        timesteps = timesteps_bayes
        
        # Compare the two approaches
        print("\n--- COMPARISON OF APPROACHES ---")
        if len(p_t_frequentist) == len(p_t_bayesian):
            correlation = np.corrcoef(p_t_frequentist, p_t_bayesian)[0, 1]
            print(f"Correlation between frequentist and Bayesian: {correlation:.3f}")
            
            # Calculate differences
            differences = p_t_frequentist - p_t_bayesian
            print(f"Mean absolute difference: {np.mean(np.abs(differences)):.4f}")
            print(f"Max absolute difference: {np.max(np.abs(differences)):.4f}")
        else:
            print("Cannot compare: different numbers of timesteps")
        
        # Compare with existing p_t_series if available
        if existing_p_t_series is not None:
            print(f"\nComparison with existing p_t_series:")
            print(f"  Calculated: {len(p_t_series)} timesteps, mean: {np.mean(p_t_series):.3f}")
            print(f"  Existing: {len(existing_p_t_series)} timesteps, mean: {np.mean(existing_p_t_series):.3f}")
            
            # Align the series for comparison
            min_length = min(len(p_t_series), len(existing_p_t_series))
            if min_length > 0:
                correlation = np.corrcoef(p_t_series[:min_length], existing_p_t_series[:min_length])[0, 1]
                print(f"  Correlation: {correlation:.3f}")
        
        # Stage 2: Infer l_t series using p_t series
        print("\n" + "=" * 40)
        print("STAGE 2: INFERRING l_t SERIES")
        print("=" * 40)
        
        l_t_dict = infer_l_t_series(
            growth_rates_data,
            p_t_series,
            population_weights,
            delta=0.05,
            n_samples=3000,  # Reduced for testing
            rolling_window=3
        )
        
        # Summary of results
        print("\n" + "=" * 40)
        print("INFERENCE RESULTS SUMMARY")
        print("=" * 40)
        print(f"Frequentist p_t series: {len(p_t_frequentist)} timesteps")
        print(f"  Range: {p_t_frequentist.min():.3f} to {p_t_frequentist.max():.3f}")
        print(f"  Mean: {np.mean(p_t_frequentist):.3f}")
        print(f"  Std: {np.std(p_t_frequentist):.3f}")
        
        print(f"\nBayesian p_t series: {len(p_t_series)} timesteps")
        print(f"  Range: {p_t_series.min():.3f} to {p_t_series.max():.3f}")
        print(f"  Mean: {np.mean(p_t_series):.3f}")
        print(f"  Std: {np.std(p_t_series):.3f}")
        
        print(f"\nl_t series: {len(l_t_dict)} timesteps")
        l_values = list(l_t_dict.values())
        print(f"  Range: {min(l_values):.3f} to {max(l_values):.3f}")
        print(f"  Mean: {np.mean(l_values):.3f}")
        print(f"  Std: {np.std(l_values):.3f}")
        
        print("\n✓ Two-stage inference completed successfully!")
        
        # Test uncertainty-aware Bayesian estimation
        print("\n" + "=" * 40)
        print("TESTING UNCERTAINTY-AWARE BAYESIAN ESTIMATION")
        print("=" * 40)
        
        # Create synthetic uncertainties for demonstration
        # In real data, these would come from measurement errors or model uncertainties
        synthetic_uncertainties = []
        for agent_growth_rates in growth_rates_data:
            # Create uncertainties that are proportional to the magnitude of growth rates
            # This simulates the common case where larger growth rates have larger uncertainties
            uncertainties = np.abs(agent_growth_rates) * 0.1 + 0.01  # 10% of magnitude + small baseline
            synthetic_uncertainties.append(uncertainties)
        
        print("Testing with synthetic uncertainties...")
        p_t_uncertainty_aware, p_t_posteriors_uncertainty, timesteps_uncertainty = bayesian_p_estimation(
            growth_rates_data,
            population_weights,
            growth_rate_uncertainties=synthetic_uncertainties,
            min_agents_per_timestep=5,
            prior_alpha=1,
            prior_beta=1
        )
        
        # Compare uncertainty-aware vs. standard Bayesian
        print(f"\nComparison of uncertainty-aware vs. standard Bayesian:")
        if len(p_t_bayesian) == len(p_t_uncertainty_aware):
            correlation = np.corrcoef(p_t_bayesian, p_t_uncertainty_aware)[0, 1]
            print(f"  Correlation: {correlation:.3f}")
            
            differences = p_t_uncertainty_aware - p_t_bayesian
            print(f"  Mean absolute difference: {np.mean(np.abs(differences)):.4f}")
            print(f"  Max absolute difference: {np.max(np.abs(differences)):.4f}")
        
        # Save results for further analysis
        results = {
            'p_t_series': p_t_series,  # Bayesian results (more robust)
            'p_t_frequentist': p_t_frequentist,
            'p_t_posteriors': p_t_posteriors,
            'p_t_uncertainty_aware': p_t_uncertainty_aware,
            'p_t_posteriors_uncertainty': p_t_posteriors_uncertainty,
            'l_t_dict': l_t_dict,
            'timesteps': timesteps,
            'n_agents': len(growth_rates_data)
        }
        
        # Save results to file
        output_file = "vinference_test_results.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✓ Results saved to {output_file}")
        
    except Exception as e:
        print(f"\n✗ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return

def main():
    """
    Main function to run the test.
    """
    print("=" * 60)
    print("VINFERENCE TEST SCRIPT")
    print("=" * 60)
    
    # Check available dummy data files
    available_files = [
        "dummy_data_kelly_betting_static_x.pkl",
        "dummy_data_kelly_betting_static_xSIMPLE.pkl"
    ]
    
    print("\nAvailable dummy data files:")
    for i, filename in enumerate(available_files):
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"  {i+1}. {filename} ({file_size:.1f} MB)")
        else:
            print(f"  {i+1}. {filename} (not found)")
    
    # Try to use the main dummy data file first
    if os.path.exists("dummy_data_kelly_betting_static_x.pkl"):
        print(f"\nUsing main dummy data file...")
        test_vinference_with_dummy_data("dummy_data_kelly_betting_static_x.pkl")
    elif os.path.exists("dummy_data_kelly_betting_static_xSIMPLE.pkl"):
        print(f"\nUsing simple dummy data file...")
        test_vinference_with_dummy_data("dummy_data_kelly_betting_static_xSIMPLE.pkl")
    else:
        print("\n✗ No dummy data files found!")
        print("Please ensure dummy data files are in the current directory.")
        return
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main() 