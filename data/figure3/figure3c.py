#!/usr/bin/env python3
"""
Figure 3C: L vs P Scatter Plot for All Cities >250k Population
===============================================================

This script runs the l inference for all cities with population over 250k
and creates a scatter plot of l vs p for each city.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import sys
from pathlib import Path

# Add the validation directory to the path to import SSM functions
sys.path.append('../validation/validation_dynamic_p_l')
from ssm_model import fit_l_cross_sectional

# Set matplotlib parameters for publication quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'ytick.labelsize': 12,
    'xtick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def load_cbsa_data():
    """Load the CBSA ACS data."""
    data_path = "../figure1/cbsa_acs_data.pkl"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Successfully loaded CBSA ACS data from {data_path}")
        print(f"   Found {len(data)} CBSAs")
        return data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def filter_cities_by_population(data, min_population=250000):
    """Filter cities to only include those with population > min_population."""
    print(f"\nFiltering cities with population > {min_population:,}...")
    
    large_cities = {}
    
    for cbsa_name, cbsa_data in data.items():
        try:
            # Check if this is a DataFrame with population data
            if isinstance(cbsa_data, pd.DataFrame) and 'population' in cbsa_data.columns:
                # Get the most recent year's population
                max_year = cbsa_data['year'].max()
                min_year = cbsa_data['year'].min()
                recent_data = cbsa_data[cbsa_data['year'] == max_year]
                
                if len(recent_data) > 0:
                    total_population = recent_data['population'].sum()
                    
                    if total_population > min_population:
                        large_cities[cbsa_name] = {
                            'data': cbsa_data,
                            'population': total_population,
                            'year': max_year,
                            'min_year': min_year,
                            'max_year': max_year
                        }
                        print(f"  ✅ {cbsa_name}: {total_population:,} people ({min_year}-{max_year})")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {cbsa_name}: {e}")
            continue
    
    print(f"\n✅ Found {len(large_cities)} cities with population > {min_population:,}")
    return large_cities

def calculate_growth_rates_and_p(city_data, target_year=2015):
    """Calculate growth rates, p value, and population weights for a specific city and year."""
    try:
        # Get data for the target year and previous year
        current_data = city_data[city_data['year'] == target_year]
        previous_data = city_data[city_data['year'] == target_year - 1]
        
        if len(current_data) == 0 or len(previous_data) == 0:
            print(f"    No data for {target_year} or {target_year-1}")
            print(f"    Available years: {sorted(city_data['year'].unique())}")
            return None, None, None, None
        
        # Merge current and previous year data
        merged_data = current_data.merge(previous_data, on='block_group_fips', suffixes=('_current', '_previous'))
        
        if len(merged_data) == 0:
            print(f"    No matching block groups between {target_year} and {target_year-1}")
            return None, None, None, None
        
        # Calculate growth rates
        growth_rates = np.log(merged_data['mean_income_current'] / merged_data['mean_income_previous'])
        
        # Filter out invalid growth rates
        valid_mask = np.isfinite(growth_rates)
        growth_rates_clean = growth_rates[valid_mask]
        merged_data_clean = merged_data[valid_mask]
        
        if len(growth_rates_clean) < 5:  # Need minimum data points
            print(f"    Only {len(growth_rates_clean)} valid growth rates (need >= 5)")
            return None, None, None, None
        
        # Calculate population-weighted p value
        total_pop = merged_data_clean['population_current'].sum()
        positive_mask = growth_rates_clean > 0
        positive_pop = merged_data_clean.loc[positive_mask, 'population_current'].sum()
        
        if total_pop > 0:
            p_value = positive_pop / total_pop
            population_weights = merged_data_clean['population_current'].values
            return growth_rates_clean, p_value, len(growth_rates_clean), population_weights
        else:
            print(f"    Total population is 0")
            return None, None, None, None
            
    except Exception as e:
        print(f"    Error calculating growth rates: {e}")
        return None, None, None, None

def run_l_inference_for_city(growth_rates, p_value, population_weights, city_name):
    """Run l inference for a single city."""
    try:
        # Run cross-sectional l inference with population weights
        idata = fit_l_cross_sectional(
            y_values_t=growth_rates,
            p_hat_t=p_value,
            population_weights=population_weights,
            delta=0.05,  # Sub-optimality offset
            n_samples=4000  # Reduced for speed
        )
        
        # Extract posterior samples
        l_posterior_samples = idata.posterior["l"].values.flatten()
        
        # Calculate point estimate and uncertainty
        l_est = np.mean(l_posterior_samples)
        l_std = np.std(l_posterior_samples)
        l_ci_95 = np.percentile(l_posterior_samples, [2.5, 97.5])
        
        return {
            'l_est': l_est,
            'l_std': l_std,
            'l_ci_95_lower': l_ci_95[0],
            'l_ci_95_upper': l_ci_95[1],
            'n_block_groups': len(growth_rates)
        }
        
    except Exception as e:
        print(f"    ❌ Failed to estimate l: {e}")
        return None

def process_all_cities(large_cities, target_year=2015):
    """Process all large cities to get l and p values."""
    print(f"\nProcessing all cities for year {target_year}...")
    
    results = []
    
    for city_name, city_info in large_cities.items():
        print(f"\nProcessing {city_name}...")
        
        # Calculate growth rates and p value
        growth_rates, p_value, n_block_groups, population_weights = calculate_growth_rates_and_p(
            city_info['data'], target_year
        )
        
        if growth_rates is not None and p_value is not None:
            print(f"  ✅ {n_block_groups} block groups, p = {p_value:.3f}")
            
            # Run l inference
            l_result = run_l_inference_for_city(growth_rates, p_value, population_weights, city_name)
            
            if l_result is not None:
                results.append({
                    'city_name': city_name,
                    'population': city_info['population'],
                    'p_value': p_value,
                    'l_est': l_result['l_est'],
                    'l_std': l_result['l_std'],
                    'l_ci_95_lower': l_result['l_ci_95_lower'],
                    'l_ci_95_upper': l_result['l_ci_95_upper'],
                    'n_block_groups': l_result['n_block_groups']
                })
                
                print(f"  ✅ l_est = {l_result['l_est']:.3f} ± {l_result['l_std']:.3f}")
            else:
                print(f"  ❌ L inference failed")
        else:
            print(f"  ❌ Insufficient data for growth rate calculation")
    
    return results

def create_scatter_plot(results, save_path="figure3"):
    """Create scatter plot of l vs p for all cities."""
    print(f"\nCreating scatter plot of l vs p...")
    
    if not results:
        print("❌ No results to plot")
        return None
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Calculate information for each city
    def calculate_information(l_val, p_val):
        """Calculate information using the formula I = ln(l) + p*ln(p) + (1-p)*ln((1-p)/(l-1))"""
        try:
            if l_val <= 1 or p_val <= 0 or p_val >= 1:
                return np.nan
            
            info = np.log(l_val) + p_val * np.log(p_val) + (1 - p_val) * np.log((1 - p_val) / (l_val - 1))
            return info
        except:
            return np.nan
    
    # Add information column
    df['information'] = df.apply(lambda row: calculate_information(row['l_est'], row['p_value']), axis=1)
    
    # Filter out any NaN values
    df_clean = df.dropna(subset=['information'])
    print(f"   Calculated information for {len(df_clean)} cities (filtered from {len(df)} total)")
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: l vs p scatter (top left)
    scatter = ax1.scatter(df_clean['p_value'], df_clean['l_est'], 
                          s=df_clean['n_block_groups']/10,  # Size by number of block groups
                          c=df_clean['population'],  # Color by population
                          cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add error bars
    ax1.errorbar(df_clean['p_value'], df_clean['l_est'], 
                 yerr=df_clean['l_std'], fmt='none', color='black', alpha=0.3, capsize=3)
    
    ax1.set_xlabel('p (Fraction with Positive Growth)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('l (Multiplicity)', fontsize=14, fontweight='bold')
    ax1.set_title('L vs P for Cities >250k Population (2015)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.4, 0.6)
    ax1.set_ylim(1.5, 2.5)
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Population', fontsize=12, fontweight='bold')
    
    # Calculate empirical growth rates for each city
    def calculate_empirical_growth_rate(city_data, target_year=2015):
        """Calculate empirical growth rate from population-weighted average income"""
        try:
            # Get data for the target year and previous year
            current_data = city_data[city_data['year'] == target_year]
            previous_data = city_data[city_data['year'] == target_year - 1]
            
            if len(current_data) == 0 or len(previous_data) == 0:
                return np.nan
            
            # Calculate population-weighted average income for each year
            current_weighted_avg = (current_data['mean_income'] * current_data['population']).sum() / current_data['population'].sum()
            previous_weighted_avg = (previous_data['mean_income'] * previous_data['population']).sum() / previous_data['population'].sum()
            
            # Calculate log growth rate
            if previous_weighted_avg > 0 and current_weighted_avg > 0:
                growth_rate = np.log(current_weighted_avg / previous_weighted_avg)
                return growth_rate
            else:
                return np.nan
                
        except Exception as e:
            return np.nan
    
    # We need to load the full city data to calculate empirical growth rates
    print("   Loading full city data to calculate empirical growth rates...")
    
    # Load CBSA data
    data = load_cbsa_data()
    if data is None:
        print("❌ Could not load CBSA data for empirical growth rate calculation")
        return None
    
    # Calculate empirical growth rates for each city
    empirical_growth_rates = {}
    for city_name in df_clean['city_name'].unique():
        if city_name in data:
            city_data = data[city_name]
            growth_rate = calculate_empirical_growth_rate(city_data, target_year=2015)
            empirical_growth_rates[city_name] = growth_rate
    
    # Add empirical growth rate column
    df_clean['empirical_growth_rate'] = df_clean['city_name'].map(empirical_growth_rates)
    
    # Filter out any NaN values in empirical growth rate
    df_clean = df_clean.dropna(subset=['empirical_growth_rate', 'information'])
    print(f"   Final data for plotting: {len(df_clean)} cities with valid empirical growth rates")
    
    # Plot 2: Empirical Growth Rate vs Information (top right)
    scatter2 = ax2.scatter(df_clean['information'], df_clean['empirical_growth_rate'], 
                           s=df_clean['n_block_groups']/10,  # Size by number of block groups
                           c=df_clean['p_value'],  # Color by p value
                           cmap='plasma', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel('Information I = ln(l) + p*ln(p) + (1-p)*ln((1-p)/(l-1))', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Empirical Growth Rate (Population-Weighted)', fontsize=14, fontweight='bold')
    ax2.set_title('Empirical Growth Rate vs Information (2015)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('p value', fontsize=12, fontweight='bold')
    
    # Plot 3: p vs Empirical Growth Rate (bottom left)
    scatter3 = ax3.scatter(df_clean['p_value'], df_clean['empirical_growth_rate'], 
                           s=df_clean['n_block_groups']/10,  # Size by number of block groups
                           c=df_clean['l_est'],  # Color by l value
                           cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax3.set_xlabel('p (Fraction with Positive Growth)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Empirical Growth Rate (Population-Weighted)', fontsize=14, fontweight='bold')
    ax3.set_title('P vs Empirical Growth Rate (2015)', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0.4, 0.6)
    
    # Add colorbar
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('l value', fontsize=12, fontweight='bold')
    
    # Plot 4: l vs Empirical Growth Rate (bottom right)
    scatter4 = ax4.scatter(df_clean['l_est'], df_clean['empirical_growth_rate'], 
                           s=df_clean['n_block_groups']/10,  # Size by number of block groups
                           c=df_clean['p_value'],  # Color by p value
                           cmap='plasma', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add error bars for l
    ax4.errorbar(df_clean['l_est'], df_clean['empirical_growth_rate'], 
                 xerr=df_clean['l_std'], fmt='none', color='black', alpha=0.3, capsize=3)
    
    ax4.set_xlabel('l (Multiplicity)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Empirical Growth Rate (Population-Weighted)', fontsize=14, fontweight='bold')
    ax4.set_title('L vs Empirical Growth Rate (2015)', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(1.5, 2.5)
    
    # Add colorbar
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('p value', fontsize=12, fontweight='bold')
    
    # Print some statistics about the information and empirical growth rate
    print(f"   Information range: {df_clean['information'].min():.3f} to {df_clean['information'].max():.3f}")
    print(f"   Empirical growth rate range: {df_clean['empirical_growth_rate'].min():.3f} to {df_clean['empirical_growth_rate'].max():.3f}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(save_path, exist_ok=True)
    pdf_path = os.path.join(save_path, "figure3c_l_vs_p_scatter.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"✓ Figure PDF saved to: {pdf_path}")
    
    png_path = os.path.join(save_path, "figure3c_l_vs_p_scatter.png")
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"✓ Figure PNG saved to: {png_path}")
    
    return fig

def check_existing_data(save_path="figure3"):
    """Check if the city results data already exists and load it if it does."""
    csv_path = os.path.join(save_path, "figure3c_city_results.csv")
    pickle_path = os.path.join(save_path, "figure3c_city_results.pkl")
    
    if os.path.exists(csv_path) and os.path.exists(pickle_path):
        print(f"✅ Found existing data files:")
        print(f"   CSV: {csv_path}")
        print(f"   Pickle: {pickle_path}")
        
        try:
            # Load from CSV for easy viewing
            df = pd.read_csv(csv_path)
            print(f"   Loaded {len(df)} cities from existing data")
            
            # Also load pickle for full data
            with open(pickle_path, 'rb') as f:
                results = pickle.load(f)
            
            print(f"   Data summary:")
            print(f"     Total cities: {len(results)}")
            print(f"     Mean l: {df['l_est'].mean():.3f} ± {df['l_est'].std():.3f}")
            print(f"     Mean p: {df['p_value'].mean():.3f} ± {df['p_value'].std():.3f}")
            print(f"     Population range: {df['population'].min():,.0f} to {df['population'].max():,.0f}")
            
            return results, df
            
        except Exception as e:
            print(f"   ❌ Error loading existing data: {e}")
            return None, None
    
    print("❌ No existing data found. Will run full analysis.")
    return None, None

def save_results(results, save_path="figure3"):
    """Save the results to CSV and pickle files."""
    print(f"\nSaving results...")
    
    if not results:
        print("❌ No results to save")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save as CSV
    csv_path = os.path.join(save_path, "figure3c_city_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Results CSV saved to: {csv_path}")
    
    # Save as pickle
    pickle_path = os.path.join(save_path, "figure3c_city_results.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results pickle saved to: {pickle_path}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total cities processed: {len(results)}")
    print(f"  Mean l: {df['l_est'].mean():.3f} ± {df['l_est'].std():.3f}")
    print(f"  Mean p: {df['p_value'].mean():.3f} ± {df['p_value'].std():.3f}")
    print(f"  Population range: {df['population'].min():,.0f} to {df['population'].max():,.0f}")
    
    return df

def main():
    """Main function to create Figure 3C."""
    print("=" * 60)
    print("FIGURE 3C: L VS P SCATTER PLOT FOR ALL CITIES >250K POPULATION")
    print("=" * 60)
    
    # First check if data already exists
    results, df = check_existing_data()
    
    if results is None:
        # No existing data, run full analysis
        print("\nRunning full analysis...")
        
        # Load CBSA data
        data = load_cbsa_data()
        if data is None:
            return
        
        # Filter cities by population
        large_cities = filter_cities_by_population(data, min_population=250000)
        if not large_cities:
            print("❌ No large cities found")
            return
        
        # Process all cities
        results = process_all_cities(large_cities, target_year=2015)
        if not results:
            print("❌ No results obtained")
            return
        
        # Save results
        df = save_results(results)
        if df is None:
            print("❌ Failed to save results")
            return
    else:
        print("\nUsing existing data for visualization...")
    
    # Create visualization (regardless of whether we loaded or generated data)
    fig = create_scatter_plot(results)
    
    print("\n" + "=" * 60)
    print("FIGURE 3C COMPLETE")
    print("=" * 60)
    print("Files created:")
    print("  - figure3/figure3c_l_vs_p_scatter.pdf")
    print("  - figure3/figure3c_l_vs_p_scatter.png")
    print("  - figure3/figure3c_city_results.csv")
    print("  - figure3/figure3c_city_results.pkl")
    print("\nThe figure shows:")
    print("  - Left: L vs P scatter plot with population-based coloring")
    print("  - Right: Mean Growth Rate vs Information scatter plot with p-value coloring")
    print("  - Point sizes indicate number of block groups")
    print("  - Error bars show uncertainty in l estimates")

if __name__ == "__main__":
    main() 