#!/usr/bin/env python3
"""
Process ACS data to compute growth rates and p-values with uncertainties.
Reads cbsa_acs_data.pkl and outputs processed_acs_data.pkl with error bars.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple

def calculate_growth_rate_with_uncertainty(current_data: pd.DataFrame, previous_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate growth rate with uncertainty using error propagation.
    
    Args:
        current_data: DataFrame with current year data (mean_income, income_moe, population, population_moe)
        previous_data: DataFrame with previous year data (mean_income, income_moe, population, population_moe)
    
    Returns:
        DataFrame with growth_rate and growth_rate_moe columns
    """
    # Merge current and previous data by block_group_fips
    merged = current_data.merge(previous_data, on='block_group_fips', suffixes=('_current', '_previous'))
    
    # Calculate log growth rate
    merged['growth_rate'] = np.log(merged['mean_income_current']) - np.log(merged['mean_income_previous'])
    
    # Calculate uncertainty in log growth rate using the derived formula
    # MOE_g = sqrt( (MOE_x / x)^2 + (MOE_y / y)^2 )
    relative_moe_current_sq = (merged['income_moe_current'] / merged['mean_income_current']) ** 2
    relative_moe_previous_sq = (merged['income_moe_previous'] / merged['mean_income_previous']) ** 2
    
    merged['growth_rate_moe'] = np.sqrt(relative_moe_current_sq + relative_moe_previous_sq)
    
    # Calculate coefficient of variation for growth rate
    # CV = (SE / mean) * 100 = (MOE / 1.645 / mean) * 100
    merged['growth_rate_cv'] = (merged['growth_rate_moe'] / 1.645) / np.abs(merged['growth_rate']) * 100
    
    # Filter out infinite or NaN values
    merged = merged[np.isfinite(merged['growth_rate'])]
    
    return merged[['block_group_fips', 'growth_rate', 'growth_rate_moe', 'growth_rate_cv']]

def calculate_population_weighted_growth_rate_with_uncertainty(growth_data: pd.DataFrame) -> Dict:
    """
    Calculate population-weighted growth rate with uncertainty.
    
    Args:
        growth_data: DataFrame with growth_rate, growth_rate_moe, and population columns
    
    Returns:
        Dictionary with weighted_mean, weighted_moe, and weighted_cv
    """
    # Population-weighted mean growth rate
    total_population = growth_data['population'].sum()
    weighted_mean = (growth_data['growth_rate'] * growth_data['population']).sum() / total_population
    
    # Population-weighted uncertainty
    # For weighted mean f = Σ(w_i * x_i) / Σ(w_i), uncertainty is:
    # σ_f = sqrt(Σ(w_i² * σ_i²)) / Σ(w_i)
    normalized_weights = growth_data['population'] / total_population
    weighted_variance = ((normalized_weights ** 2) * (growth_data['growth_rate_moe'] / 1.645) ** 2).sum()
    weighted_moe = np.sqrt(weighted_variance) * 1.645
    
    # Weighted coefficient of variation
    weighted_cv = (weighted_moe / 1.645) / np.abs(weighted_mean) * 100
    
    return {
        'weighted_mean': weighted_mean,
        'weighted_moe': weighted_moe,
        'weighted_cv': weighted_cv,
        'total_population': total_population
    }

def calculate_p_with_uncertainty(growth_data: pd.DataFrame) -> Dict:
    """
    Calculate p (fraction of positive growth rates) with uncertainty.
    
    Args:
        growth_data: DataFrame with growth_rate, growth_rate_moe, and population columns
    
    Returns:
        Dictionary with p_value, p_moe, and p_cv
    """
    # Calculate p as fraction of positive growth rates (weighted by population)
    positive_mask = growth_data['growth_rate'] > 0
    positive_population = growth_data.loc[positive_mask, 'population'].sum()
    total_population = growth_data['population'].sum()
    
    p_value = positive_population / total_population
    
    # Uncertainty in p using binomial proportion uncertainty
    # For a proportion p = x/n, the standard error is sqrt(p(1-p)/n)
    # But since we have weighted populations, we need to account for that
    
    # Calculate effective sample size (total population)
    n_effective = total_population
    
    # Standard error for proportion
    p_se = np.sqrt(p_value * (1 - p_value) / n_effective)
    p_moe = p_se * 1.645  # Convert to 90% confidence MOE
    
    # Coefficient of variation
    p_cv = (p_se / p_value) * 100 if p_value > 0 else np.nan
    
    return {
        'p_value': p_value,
        'p_moe': p_moe,
        'p_cv': p_cv,
        'positive_population': positive_population,
        'total_population': total_population
    }

def process_cbsa_data(cbsa_name: str, cbsa_data: pd.DataFrame) -> Dict:
    """
    Process a single CBSA to compute growth rates and p-values with uncertainties.
    
    Args:
        cbsa_name: Name of the CBSA
        cbsa_data: DataFrame with ACS data for the CBSA
    
    Returns:
        Dictionary containing processed data and summary statistics
    """
    print(f"Processing CBSA: {cbsa_name}")
    
    # --- Data Validation Step ---
    required_columns = ['year', 'block_group_fips', 'mean_income', 'income_moe', 'population']
    missing_cols = [col for col in required_columns if col not in cbsa_data.columns]
    if missing_cols:
        error_msg = (
            f"Input data for {cbsa_name} is missing required columns: {missing_cols}. "
            f"Please re-run the full data preparation pipeline to regenerate the data files from scratch. "
            f"Start by running 'data_prep.py'."
        )
        raise ValueError(error_msg)
    # --------------------------
    
    # Get unique years and sort them
    years = sorted(cbsa_data['year'].unique())
    print(f"  Years: {years[0]} - {years[-1]} ({len(years)} years)")
    
    # Store results
    growth_results = []
    p_time_series = []
    
    # Calculate growth rates between consecutive years
    for i in range(1, len(years)):
        current_year = years[i]
        previous_year = years[i-1]
        
        # Get data for current and previous years
        current_data = cbsa_data[cbsa_data['year'] == current_year].copy()
        previous_data = cbsa_data[cbsa_data['year'] == previous_year].copy()
        
        # Calculate growth rates with uncertainties
        growth_data = calculate_growth_rate_with_uncertainty(current_data, previous_data)
        
        # Add year and population information
        growth_data['year'] = current_year
        growth_data = growth_data.merge(
            current_data[['block_group_fips', 'population']], 
            on='block_group_fips'
        )
        
        # Calculate population-weighted statistics
        weighted_stats = calculate_population_weighted_growth_rate_with_uncertainty(growth_data)
        p_stats = calculate_p_with_uncertainty(growth_data)
        
        # Store growth data
        growth_results.append({
            'year': current_year,
            'growth_data': growth_data,
            'weighted_stats': weighted_stats,
            'p_stats': p_stats
        })
        
        # Store p time series
        p_time_series.append({
            'year': current_year,
            'p_value': p_stats['p_value'],
            'p_moe': p_stats['p_moe'],
            'p_cv': p_stats['p_cv'],
            'mean_growth_rate': weighted_stats['weighted_mean'],
            'growth_rate_moe': weighted_stats['weighted_moe'],
            'growth_rate_cv': weighted_stats['weighted_cv'],
            'total_population': weighted_stats['total_population']
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(p_time_series)
    
    print(f"  Processed {len(growth_results)} year pairs")
    print(f"  Final p-value: {p_time_series[-1]['p_value']:.4f} ± {p_time_series[-1]['p_moe']:.4f}")
    print(f"  Final growth rate: {p_time_series[-1]['mean_growth_rate']:.4f} ± {p_time_series[-1]['growth_rate_moe']:.4f}")
    
    return {
        'cbsa_name': cbsa_name,
        'summary': summary_df,
        'detailed_growth_results': growth_results,
        'years': years
    }

def main():
    """Main function to process all ACS data."""
    
    # Try to use ZIP-matched data first, fall back to raw ACS data
    zip_matched_file = 'zip_matching/blockgroups_with_zips_temporal.pkl'
    raw_acs_file = 'cbsa_acs_data.pkl'
    output_file = 'processed_acs_data.pkl'
    
    # Check which input file exists
    if os.path.exists(zip_matched_file):
        input_file = zip_matched_file
        print(f"Using ZIP-matched data: {input_file}")
    elif os.path.exists(raw_acs_file):
        input_file = raw_acs_file
        print(f"Using raw ACS data: {input_file}")
        print("Note: ZIP code information will not be available.")
    else:
        print(f"Error: Neither input file found!")
        print(f"Expected: {zip_matched_file} or {raw_acs_file}")
        print("Please run acs_data_retrieval.py and match_zips.py first.")
        return
    
    print("Loading ACS data...")
    with open(input_file, 'rb') as f:
        cbsa_data = pickle.load(f)
    
    print(f"Loaded data for {len(cbsa_data)} CBSAs")
    
    # Process each CBSA
    processed_data = {}
    all_summaries = []
    
    print(f"\nProcessing CBSAs...")
    for cbsa_name, cbsa_df in cbsa_data.items():
        print(f"  Processing {cbsa_name}...")
        try:
            result = process_cbsa_data(cbsa_name, cbsa_df)
            processed_data[cbsa_name] = result
            all_summaries.append(result['summary'])
            print(f"    ✅ Successfully processed {cbsa_name}")
        except Exception as e:
            print(f"    ❌ Error processing {cbsa_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nSuccessfully processed {len(processed_data)} CBSAs")
    print(f"Generated {len(all_summaries)} summaries")
    
    # Create combined summary across all CBSAs
    if not all_summaries:
        print("❌ No CBSAs were successfully processed. Cannot create combined summary.")
        return
    
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary['cbsa_name'] = [result['cbsa_name'] for result in processed_data.values() for _ in range(len(result['summary']))]
    
    # Add combined summary to processed data
    processed_data['_combined_summary'] = combined_summary
    
    # Save processed data
    print(f"\nSaving processed data to '{output_file}'...")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("Processing complete!")
    print(f"Processed {len(processed_data) - 1} CBSAs")  # -1 for combined summary
    
    # Check if ZIP codes are available
    sample_cbsa = next(iter(processed_data.values()))
    if 'closest_zip' in sample_cbsa['summary'].columns:
        print("✅ ZIP code information included in processed data")
    else:
        print("ℹ️  ZIP code information not available in processed data")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    # Filter out the combined summary key
    cbsa_keys = [k for k in processed_data.keys() if k != '_combined_summary']
    
    for cbsa_name in cbsa_keys:
        summary = processed_data[cbsa_name]['summary']
        if not summary.empty:
            latest = summary.iloc[-1]
            print(f"{cbsa_name}:")
            print(f"  Latest p: {latest['p_value']:.4f} ± {latest['p_moe']:.4f}")
            print(f"  Latest growth: {latest['mean_growth_rate']:.4f} ± {latest['growth_rate_moe']:.4f}")
            print(f"  Years: {len(summary)}")
            print()

if __name__ == "__main__":
    main() 