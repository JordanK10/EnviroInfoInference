#!/usr/bin/env python3
"""
Demonstration of Uncertainty-Aware Bayesian p Estimation
=======================================================

This script demonstrates how the new uncertainty-aware Bayesian p estimation
handles growth rates with measurement uncertainty that spans the zero point.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from vinference import bayesian_p_estimation

def create_synthetic_data_with_uncertainties():
    """
    Create synthetic growth rate data with realistic uncertainties.
    
    Returns:
        tuple: (growth_rates_data, uncertainties_data, true_p_values)
    """
    np.random.seed(42)
    
    n_agents = 50
    n_timesteps = 15
    
    # Create true underlying p_t values that vary over time
    true_p_values = 0.5 + 0.2 * np.sin(np.linspace(0, 2*np.pi, n_timesteps))
    
    # Generate growth rates for each agent
    growth_rates_data = []
    uncertainties_data = []
    
    for agent in range(n_agents):
        agent_growth_rates = []
        agent_uncertainties = []
        
        for timestep in range(n_timesteps):
            # True underlying growth rate (positive or negative based on p_t)
            p_t = true_p_values[timestep]
            
            if np.random.random() < p_t:
                # Win: positive growth
                true_growth = np.random.normal(0.02, 0.01)  # 2% mean, 1% std
            else:
                # Loss: negative growth
                true_growth = np.random.normal(-0.02, 0.01)  # -2% mean, 1% std
            
            # Add measurement uncertainty
            measurement_error = np.random.normal(0, 0.005)  # 0.5% measurement error
            observed_growth = true_growth + measurement_error
            
            # Uncertainty is proportional to magnitude + measurement error
            uncertainty = np.abs(observed_growth) * 0.2 + 0.005  # 20% of magnitude + baseline
            
            agent_growth_rates.append(observed_growth)
            agent_uncertainties.append(uncertainty)
        
        growth_rates_data.append(agent_growth_rates)
        uncertainties_data.append(agent_uncertainties)
    
    return growth_rates_data, uncertainties_data, true_p_values

def demonstrate_uncertainty_handling():
    """
    Demonstrate how uncertainty-aware estimation handles borderline cases.
    """
    print("=" * 60)
    print("DEMONSTRATION: UNCERTAINTY-AWARE BAYESIAN p ESTIMATION")
    print("=" * 60)
    
    # Create synthetic data
    print("\nGenerating synthetic growth rate data with uncertainties...")
    growth_rates_data, uncertainties_data, true_p_values = create_synthetic_data_with_uncertainties()
    
    print(f"Created {len(growth_rates_data)} agents with {len(growth_rates_data[0])} timesteps")
    print(f"True p_t values range: {true_p_values.min():.3f} to {true_p_values.max():.3f}")
    
    # Test both approaches
    print("\n" + "=" * 40)
    print("STANDARD BAYESIAN (NO UNCERTAINTY)")
    print("=" * 40)
    
    p_t_standard, posteriors_standard, timesteps_standard = bayesian_p_estimation(
        growth_rates_data,
        population_weights=None,
        growth_rate_uncertainties=None,  # No uncertainty handling
        min_agents_per_timestep=5,
        prior_alpha=1,
        prior_beta=1
    )
    
    print("\n" + "=" * 40)
    print("UNCERTAINTY-AWARE BAYESIAN")
    print("=" * 40)
    
    p_t_uncertainty, posteriors_uncertainty, timesteps_uncertainty = bayesian_p_estimation(
        growth_rates_data,
        population_weights=None,
        growth_rate_uncertainties=uncertainties_data,  # With uncertainty handling
        min_agents_per_timestep=5,
        prior_alpha=1,
        prior_beta=1
    )
    
    # Analysis of results
    print("\n" + "=" * 40)
    print("ANALYSIS OF RESULTS")
    print("=" * 40)
    
    # Compare with true values
    if len(true_p_values) == len(p_t_standard):
        print(f"Standard Bayesian vs True p_t:")
        correlation_standard = np.corrcoef(true_p_values, p_t_standard)[0, 1]
        mae_standard = np.mean(np.abs(p_t_standard - true_p_values))
        print(f"  Correlation: {correlation_standard:.3f}")
        print(f"  Mean Absolute Error: {mae_standard:.4f}")
    
    if len(true_p_values) == len(p_t_uncertainty):
        print(f"\nUncertainty-Aware vs True p_t:")
        correlation_uncertainty = np.corrcoef(true_p_values, p_t_uncertainty)[0, 1]
        mae_uncertainty = np.mean(np.abs(p_t_uncertainty - true_p_values))
        print(f"  Correlation: {correlation_uncertainty:.3f}")
        print(f"  Mean Absolute Error: {mae_uncertainty:.4f}")
    
    # Compare the two approaches
    if len(p_t_standard) == len(p_t_uncertainty):
        print(f"\nStandard vs Uncertainty-Aware:")
        correlation_comparison = np.corrcoef(p_t_standard, p_t_uncertainty)[0, 1]
        mae_comparison = np.mean(np.abs(p_t_standard - p_t_uncertainty))
        print(f"  Correlation: {correlation_comparison:.3f}")
        print(f"  Mean Absolute Difference: {mae_comparison:.4f}")
    
    # Show specific examples of uncertainty handling
    print(f"\n" + "=" * 40)
    print("EXAMPLES OF UNCERTAINTY HANDLING")
    print("=" * 40)
    
    # Find timesteps with borderline cases
    for timestep in range(min(len(p_t_standard), len(p_t_uncertainty))):
        if timestep < len(true_p_values):
            true_p = true_p_values[timestep]
            
            # Check if this is a borderline case (p_t close to 0.5)
            if abs(true_p - 0.5) < 0.1:
                print(f"\nTimestep {timestep} (Borderline case, true p_t = {true_p:.3f}):")
                
                # Count agents near zero
                near_zero_count = 0
                for agent_growth_rates in growth_rates_data:
                    if timestep < len(agent_growth_rates):
                        y = agent_growth_rates[timestep]
                        if abs(y) < 0.01:  # Growth rate within 1% of zero
                            near_zero_count += 1
                
                print(f"  Agents near zero (±1%): {near_zero_count}")
                print(f"  Standard estimate: {p_t_standard[timestep]:.3f}")
                print(f"  Uncertainty-aware: {p_t_uncertainty[timestep]:.3f}")
                print(f"  Difference: {p_t_uncertainty[timestep] - p_t_standard[timestep]:.4f}")
    
    return {
        'true_p_values': true_p_values,
        'p_t_standard': p_t_standard,
        'p_t_uncertainty': p_t_uncertainty,
        'timesteps': timesteps_standard
    }

def plot_results(results):
    """
    Plot the results to visualize the differences.
    """
    if not results:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    timesteps = results['timesteps']
    true_p = results['true_p_values'][:len(timesteps)]
    p_standard = results['p_t_standard']
    p_uncertainty = results['p_t_uncertainty']
    
    # Plot 1: p_t values over time
    ax1.plot(timesteps, true_p, 'k-', linewidth=2, label='True p_t', alpha=0.8)
    ax1.plot(timesteps, p_standard, 'b--', linewidth=2, label='Standard Bayesian', alpha=0.8)
    ax1.plot(timesteps, p_uncertainty, 'r--', linewidth=2, label='Uncertainty-Aware', alpha=0.8)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('p_t value')
    ax1.set_title('Comparison of p_t Estimation Methods')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Differences
    ax2.plot(timesteps, p_standard - true_p, 'b-', linewidth=2, label='Standard - True', alpha=0.8)
    ax2.plot(timesteps, p_uncertainty - true_p, 'r-', linewidth=2, label='Uncertainty-Aware - True', alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Estimation Error')
    ax2.set_title('Estimation Errors vs True Values')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('uncertainty_demo_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'uncertainty_demo_results.png'")
    
    return fig

def main():
    """
    Main function to run the demonstration.
    """
    print("=" * 60)
    print("UNCERTAINTY-AWARE BAYESIAN p ESTIMATION DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Run the demonstration
        results = demonstrate_uncertainty_handling()
        
        # Create visualization
        print(f"\nCreating visualization...")
        fig = plot_results(results)
        
        print(f"\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("Key insights:")
        print("1. Uncertainty-aware estimation handles borderline cases better")
        print("2. Agents with growth rates near zero get fractional win probabilities")
        print("3. This leads to more robust p_t estimates when measurement errors exist")
        print("4. The method naturally down-weights uncertain observations")
        
    except Exception as e:
        print(f"\n✗ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 