#!/usr/bin/env python3
"""
Test Vinference with Albuquerque Real Data
==========================================

This script tests the vinference functions using real Albuquerque CBSA ACS data
with estimated uncertainties to demonstrate the uncertainty-aware Bayesian estimation.

Author: [Your Name]
Date: [Current Date]
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vinference import frequentist_p_estimation, bayesian_p_estimation, infer_l_t_series

def load_albuquerque_extracted_data(data_file="albuquerque_extracted_data.pkl"):
    """
    Load the extracted Albuquerque data for testing.
    
    Parameters:
    -----------
    data_file : str
        Path to the extracted data file
        
    Returns:
    --------
    dict
        Loaded Albuquerque data with growth rates and uncertainties
    """
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Successfully loaded Albuquerque extracted data from {data_file}")
        return data
    except Exception as e:
        print(f"✗ Error loading Albuquerque data: {e}")
        return None

def test_vinference_with_albuquerque_data(albuquerque_data):
    """
    Test the vinference functions with real Albuquerque data.
    
    Parameters:
    -----------
    albuquerque_data : dict
        Loaded Albuquerque data with growth rates and uncertainties
    """
    print("=" * 60)
    print("TESTING VINFERENCE WITH ALBUQUERQUE REAL DATA")
    print("=" * 60)
    
    # Extract data
    growth_rates_data = albuquerque_data['growth_rates_data']
    population_weights = albuquerque_data['population_weights']
    years = albuquerque_data['years']
    uncertainties = albuquerque_data['uncertainties']
    
    print(f"Data summary:")
    print(f"  Block groups: {len(growth_rates_data)}")
    print(f"  Years: {years[0]} to {years[-1]} ({len(years)} timesteps)")
    print(f"  Growth rate range: {np.min([np.min(gr) for gr in growth_rates_data]):.4f} to {np.max([np.max(gr) for gr in growth_rates_data]):.4f}")
    print(f"  Uncertainty methods available: {list(uncertainties.keys())}")
    
    # Test 1: Frequentist approach
    print("\n" + "=" * 40)
    print("TEST 1: FREQUENTIST p ESTIMATION")
    print("=" * 40)
    
    p_t_frequentist, timesteps_freq = frequentist_p_estimation(
        growth_rates_data,
        population_weights,
        min_agents_per_timestep=5
    )
    
    # Test 2: Standard Bayesian approach (no uncertainties)
    print("\n" + "=" * 40)
    print("TEST 2: STANDARD BAYESIAN p ESTIMATION")
    print("=" * 40)
    
    p_t_bayesian_standard, p_t_posteriors_standard, timesteps_bayes = bayesian_p_estimation(
        growth_rates_data,
        population_weights,
        growth_rate_uncertainties=None,  # No uncertainty handling
        min_agents_per_timestep=5,
        prior_alpha=1,
        prior_beta=1
    )
    
    # Test 3: Uncertainty-aware Bayesian approach
    print("\n" + "=" * 40)
    print("TEST 3: UNCERTAINTY-AWARE BAYESIAN p ESTIMATION")
    print("=" * 40)
    
    # Test with different uncertainty methods
    uncertainty_results = {}
    
    for method_name, method_uncertainties in uncertainties.items():
        print(f"\n--- Testing {method_name.upper()} uncertainty method ---")
        
        try:
            p_t_uncertainty, p_t_posteriors_uncertainty, timesteps_uncertainty = bayesian_p_estimation(
                growth_rates_data,
                population_weights,
                growth_rate_uncertainties=method_uncertainties,
                min_agents_per_timestep=5,
                prior_alpha=1,
                prior_beta=1
            )
            
            uncertainty_results[method_name] = {
                'p_t_series': p_t_uncertainty,
                'posteriors': p_t_posteriors_uncertainty,
                'timesteps': timesteps_uncertainty
            }
            
            print(f"  ✓ {method_name} method completed successfully")
            
        except Exception as e:
            print(f"  ✗ {method_name} method failed: {str(e)}")
    
    # Analysis and comparison
    print("\n" + "=" * 40)
    print("ANALYSIS AND COMPARISON")
    print("=" * 40)
    
    # Compare frequentist vs standard Bayesian
    if len(p_t_frequentist) == len(p_t_bayesian_standard):
        correlation_freq_bayes = np.corrcoef(p_t_frequentist, p_t_bayesian_standard)[0, 1]
        mae_freq_bayes = np.mean(np.abs(p_t_frequentist - p_t_bayesian_standard))
        print(f"Frequentist vs Standard Bayesian:")
        print(f"  Correlation: {correlation_freq_bayes:.3f}")
        print(f"  Mean Absolute Difference: {mae_freq_bayes:.4f}")
    
    # Compare different uncertainty methods
    if len(uncertainty_results) > 1:
        print(f"\nUncertainty Method Comparisons:")
        
        # Find the first successful method for comparison
        first_method = list(uncertainty_results.keys())[0]
        first_p_t = uncertainty_results[first_method]['p_t_series']
        
        for method_name, method_data in uncertainty_results.items():
            if method_name != first_method:
                method_p_t = method_data['p_t_series']
                if len(method_p_t) == len(first_p_t):
                    correlation = np.corrcoef(first_p_t, method_p_t)[0, 1]
                    mae = np.mean(np.abs(first_p_t - method_p_t))
                    print(f"  {first_method} vs {method_name}:")
                    print(f"    Correlation: {correlation:.3f}")
                    print(f"    Mean Absolute Difference: {mae:.4f}")
    
    # Test l_t inference with the best p_t estimates
    print("\n" + "=" * 40)
    print("TEST 4: l_t INFERENCE WITH REAL DATA")
    print("=" * 40)
    
    # Use uncertainty-aware results if available, otherwise standard Bayesian
    if uncertainty_results:
        best_method = 'empirical'  # Prefer empirical method
        if best_method in uncertainty_results:
            p_t_for_l_inference = uncertainty_results[best_method]['p_t_series']
            print(f"Using {best_method} uncertainty method for l_t inference")
        else:
            p_t_for_l_inference = list(uncertainty_results.values())[0]['p_t_series']
            print(f"Using {list(uncertainty_results.keys())[0]} uncertainty method for l_t inference")
    else:
        p_t_for_l_inference = p_t_bayesian_standard
        print("Using standard Bayesian p_t for l_t inference")
    
    try:
        l_t_dict = infer_l_t_series(
            growth_rates_data,
            p_t_for_l_inference,
            population_weights,
            delta=0.05,
            n_samples=3000,  # Reduced for real data testing
            rolling_window=3
        )
        
        print(f"✓ l_t inference completed successfully")
        print(f"  l_t estimates: {len(l_t_dict)} timesteps")
        l_values = list(l_t_dict.values())
        print(f"  l_t range: {min(l_values):.3f} to {max(l_values):.3f}")
        print(f"  l_t mean: {np.mean(l_values):.3f}")
        
    except Exception as e:
        print(f"✗ l_t inference failed: {str(e)}")
        l_t_dict = None
    
    # Save comprehensive results
    results = {
        'source': 'Albuquerque CBSA ACS data',
        'years': years,
        'n_block_groups': len(growth_rates_data),
        'p_t_frequentist': p_t_frequentist,
        'p_t_bayesian_standard': p_t_bayesian_standard,
        'p_t_uncertainty_methods': uncertainty_results,
        'l_t_dict': l_t_dict,
        'timesteps': timesteps_bayes
    }
    
    output_file = "albuquerque_vinference_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✓ Comprehensive results saved to {output_file}")
    
    return results

def create_albuquerque_visualization(results):
    """
    Create visualization of the Albuquerque vinference results.
    
    Parameters:
    -----------
    results : dict
        Results from the vinference testing
    """
    if not results:
        return None
    
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    years = results['years']
    timesteps = results['timesteps']
    
    # Plot 1: p_t comparison
    ax1 = axes[0]
    ax1.plot(timesteps, results['p_t_frequentist'], 'b-', linewidth=2, label='Frequentist', alpha=0.8)
    ax1.plot(timesteps, results['p_t_bayesian_standard'], 'r--', linewidth=2, label='Standard Bayesian', alpha=0.8)
    
    # Plot uncertainty method results
    for method_name, method_data in results['p_t_uncertainty_methods'].items():
        method_timesteps = method_data['timesteps']
        method_p_t = method_data['p_t_series']
        ax1.plot(method_timesteps, method_p_t, '--', linewidth=2, label=f'{method_name.title()} Uncertainty', alpha=0.8)
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('p_t value')
    ax1.set_title('Albuquerque p_t Estimation: Comparison of Methods')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Differences between methods
    ax2 = axes[1]
    if len(results['p_t_frequentist']) == len(results['p_t_bayesian_standard']):
        differences = results['p_t_bayesian_standard'] - results['p_t_frequentist']
        ax2.plot(timesteps, differences, 'g-', linewidth=2, label='Bayesian - Frequentist', alpha=0.8)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Difference')
        ax2.set_title('Differences Between Estimation Methods')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: l_t estimates
    ax3 = axes[2]
    if results['l_t_dict']:
        l_timesteps = sorted(results['l_t_dict'].keys())
        l_values = [results['l_t_dict'][t] for t in l_timesteps]
        ax3.plot(l_timesteps, l_values, 'purple', linewidth=2, marker='o', markersize=4, label='l_t estimates')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('l_t value')
        ax3.set_title('Albuquerque l_t Estimates')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('albuquerque_vinference_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved as 'albuquerque_vinference_results.png'")
    
    return fig

def main():
    """
    Main function to test vinference with Albuquerque data.
    """
    print("=" * 60)
    print("ALBUQUERQUE VINFERENCE TESTING")
    print("=" * 60)
    
    # Check if extracted data exists
    if not os.path.exists("albuquerque_extracted_data.pkl"):
        print("✗ Albuquerque extracted data not found!")
        print("Please run 'python examine_albuquerque_data.py' first to extract the data.")
        return
    
    # Load Albuquerque data
    albuquerque_data = load_albuquerque_extracted_data()
    if albuquerque_data is None:
        return
    
    try:
        # Test vinference functions
        results = test_vinference_with_albuquerque_data(albuquerque_data)
        
        # Create visualization
        fig = create_albuquerque_visualization(results)
        
        print("\n" + "=" * 60)
        print("ALBUQUERQUE VINFERENCE TESTING COMPLETE")
        print("=" * 60)
        print("Key findings:")
        print("1. Real data provides realistic test of uncertainty handling")
        print("2. Different uncertainty estimation methods can be compared")
        print("3. Results show how the methods perform on actual urban economic data")
        print("4. Population weighting reflects real demographic patterns")
        
    except Exception as e:
        print(f"\n✗ Testing failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 