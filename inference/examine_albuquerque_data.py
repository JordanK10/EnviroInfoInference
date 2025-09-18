#!/usr/bin/env python3
"""
Examine Albuquerque CBSA ACS Data
================================

This script examines the structure of the Albuquerque income data to understand
how to extract growth rates and uncertainties for testing the vinference functions.
It now uses precomputed data from the urbandata pipeline instead of recalculating.

Author: [Your Name]
Date: [Current Date]
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_albuquerque_data(data_path="cbsa_acs_data.pkl"):
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
            for cbsa_name in list(data.keys())[:10]:  # Show first 10
                print(f"  - {cbsa_name}")
            if len(data) > 10:
                print(f"  ... and {len(data) - 10} more")
            return None
        
        return albuquerque_data
        
    except Exception as e:
        print(f"✗ Error loading CBSA ACS data: {e}")
        return None

def load_urbandata_processed_data(urbandata_path="../urbandata/data"):
    """
    Load precomputed growth rates and uncertainties from urbandata pipeline.
    
    Parameters:
    -----------
    urbandata_path : str
        Path to urbandata directory
        
    Returns:
    --------
    dict or None
        Processed data with growth rates and uncertainties, or None if not found
    """
    print(f"\nLooking for precomputed data in urbandata directory: {urbandata_path}")
    
    # Check for processed ACS data
    processed_file = os.path.join(urbandata_path, "processed_acs_data.pkl")
    if os.path.exists(processed_file):
        print(f"✓ Found processed ACS data: {processed_file}")
        try:
            with open(processed_file, 'rb') as f:
                processed_data = pickle.load(f)
            print(f"  Contains data for {len(processed_data)} CBSAs")
            return processed_data
        except Exception as e:
            print(f"✗ Error loading processed data: {e}")
    
    # Check for ZIP-matched data
    zip_matched_file = os.path.join(urbandata_path, "zip_matching/blockgroups_with_zips_temporal.pkl")
    if os.path.exists(zip_matched_file):
        print(f"✓ Found ZIP-matched data: {zip_matched_file}")
        try:
            with open(zip_matched_file, 'rb') as f:
                zip_data = pickle.load(f)
            print(f"  Contains data for {len(zip_data)} CBSAs")
            return zip_data
        except Exception as e:
            print(f"✗ Error loading ZIP-matched data: {e}")
    
    print("✗ No precomputed data found in urbandata directory")
    return None

def examine_data_structure(albuquerque_data):
    """
    Examine the structure of the Albuquerque data to understand its format.
    
    Parameters:
    -----------
    albuquerque_data : dict or DataFrame
        The loaded Albuquerque data
    """
    print("\n" + "=" * 50)
    print("EXAMINING ALBUQUERQUE DATA STRUCTURE")
    print("=" * 50)
    
    if isinstance(albuquerque_data, pd.DataFrame):
        print("Data is a pandas DataFrame")
        print(f"Shape: {albuquerque_data.shape}")
        print(f"Columns: {list(albuquerque_data.columns)}")
        print(f"Data types:\n{albuquerque_data.dtypes}")
        
        # Show sample data
        print(f"\nFirst few rows:")
        print(albuquerque_data.head())
        
        # Check for missing values
        print(f"\nMissing values per column:")
        print(albuquerque_data.isnull().sum())
        
        # Check unique values in key columns
        if 'year' in albuquerque_data.columns:
            print(f"\nYears available: {sorted(albuquerque_data['year'].unique())}")
        if 'block_group_fips' in albuquerque_data.columns:
            print(f"Number of block groups: {albuquerque_data['block_group_fips'].nunique()}")
        if 'mean_income' in albuquerque_data.columns:
            print(f"Income range: ${albuquerque_data['mean_income'].min():,.0f} to ${albuquerque_data['mean_income'].max():,.0f}")
        if 'population' in albuquerque_data.columns:
            print(f"Population range: {albuquerque_data['population'].min():,.0f} to {albuquerque_data['population'].max():,.0f}")
            
    else:
        print("Data is a dictionary")
        print(f"Keys: {list(albuquerque_data.keys())}")
        
        for key, value in albuquerque_data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} - {value.dtype}")
            elif isinstance(value, pd.DataFrame):
                print(f"  {key}: DataFrame {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: List with {len(value)} items")
            else:
                print(f"  {key}: {type(value)}")

def extract_from_processed_urbandata(processed_data, cbsa_name):
    """
    Extract growth rates and uncertainties from processed urbandata.
    
    Parameters:
    -----------
    processed_data : dict
        Processed data from urbandata pipeline
    cbsa_name : str
        Name of the CBSA to extract data for
        
    Returns:
    --------
    tuple
        (growth_rates_data, population_weights, years, uncertainties) or (None, None, None, None)
    """
    print(f"\nExtracting data for CBSA: {cbsa_name}")
    
    if cbsa_name not in processed_data:
        print(f"✗ CBSA {cbsa_name} not found in processed data")
        available_cbsas = [k for k in processed_data.keys() if not k.startswith('_')]
        print(f"Available CBSAs: {available_cbsas[:10]}...")
        return None, None, None, None
    
    cbsa_data = processed_data[cbsa_name]
    
    # Check if this is the new processed format
    if 'detailed_growth_results' in cbsa_data:
        print("✓ Found processed format with detailed growth results")
        
        growth_results = cbsa_data['detailed_growth_results']
        years = cbsa_data['years']
        
        # Extract growth rates and uncertainties
        growth_rates_data = []
        population_weights = []
        uncertainties = []
        
        for result in growth_results:
            year = result['year']
            growth_data = result['growth_data']
            
            # Group by block group to get time series
            for bg_fips, bg_group in growth_data.groupby('block_group_fips'):
                if len(bg_group) == 1:  # Should have one row per year
                    growth_rate = bg_group['growth_rate'].iloc[0]
                    growth_moe = bg_group['growth_rate_moe'].iloc[0]
                    population = bg_group['population'].iloc[0]
                    
                    # Store data
                    growth_rates_data.append([growth_rate])
                    population_weights.append(population)
                    uncertainties.append([growth_moe])
        
        # Convert to proper format
        if growth_rates_data:
            # Transpose to get (n_agents, n_timesteps) format
            growth_rates_data = np.array(growth_rates_data).T.tolist()
            uncertainties = np.array(uncertainties).T.tolist()
            
            print(f"✓ Extracted {len(growth_rates_data[0])} block groups for {len(growth_rates_data)} timesteps")
            return growth_rates_data, population_weights, years[1:], uncertainties
        
    # Check if this is the ZIP-matched format
    elif isinstance(cbsa_data, pd.DataFrame):
        print("✓ Found ZIP-matched format")
        
        # Check if growth rates are already computed
        if 'growth_rate' in cbsa_data.columns:
            print("✓ Growth rates already computed")
            
            # Group by block group and year
            growth_rates_data = []
            population_weights = []
            uncertainties = []
            
            for bg_fips, bg_group in cbsa_data.groupby('block_group_fips'):
                bg_group = bg_group.sort_values('year')
                
                if len(bg_group) > 1:  # Need at least 2 years for growth rates
                    # Extract growth rates and uncertainties
                    growth_rates = bg_group['growth_rate'].values[1:]  # Skip first year
                    growth_moes = bg_group['growth_rate_moe'].values[1:] if 'growth_rate_moe' in bg_group.columns else np.zeros_like(growth_rates)
                    populations = bg_group['population'].values[1:]
                    
                    if np.all(np.isfinite(growth_rates)):
                        growth_rates_data.append(growth_rates)
                        population_weights.append(np.mean(populations))
                        uncertainties.append(growth_moes)
            
            if growth_rates_data:
                years = sorted(cbsa_data['year'].unique())[1:]  # Skip first year
                print(f"✓ Extracted {len(growth_rates_data)} block groups for {len(years)} timesteps")
                return growth_rates_data, population_weights, years, uncertainties
    
    print("✗ Could not extract data from processed format")
    return None, None, None, None

def extract_growth_rates_from_albuquerque(albuquerque_data, urbandata_processed=None):
    """
    Extract growth rates from Albuquerque income data.
    First tries to use precomputed data from urbandata, then falls back to calculation.
    
    Parameters:
    -----------
    albuquerque_data : dict or DataFrame
        The loaded Albuquerque data
    urbandata_processed : dict
        Precomputed data from urbandata pipeline
        
    Returns:
    --------
    tuple
        (growth_rates_data, population_weights, years) or (None, None, None) if failed
    """
    print("\n" + "=" * 50)
    print("EXTRACTING GROWTH RATES FROM ALBUQUERQUE DATA")
    print("=" * 50)
    
    # First, try to use precomputed data from urbandata
    if urbandata_processed is not None:
        print("Trying to extract from precomputed urbandata...")
        
        # Find Albuquerque in the processed data
        albuquerque_cbsa = None
        for cbsa_name in urbandata_processed.keys():
            if 'albuquerque' in cbsa_name.lower() and 'nm' in cbsa_name.lower():
                albuquerque_cbsa = cbsa_name
                break
        
        if albuquerque_cbsa:
            growth_rates_data, population_weights, years, uncertainties = extract_from_processed_urbandata(
                urbandata_processed, albuquerque_cbsa
            )
            
            if growth_rates_data is not None:
                print("✓ Successfully extracted from precomputed urbandata")
                return growth_rates_data, population_weights, years, uncertainties
        
        print("⚠ Could not extract from urbandata, falling back to calculation...")
    
    # Fallback: Calculate growth rates from income data
    print("⚠ No precomputed growth rates found, falling back to calculation from income data...")
    
    if isinstance(albuquerque_data, pd.DataFrame):
        # Handle long-format DataFrame
        expected_cols = ['year', 'block_group_fips', 'mean_income', 'population']
        if not all(col in albuquerque_data.columns for col in expected_cols):
            print(f"✗ Expected columns {expected_cols}, got {list(albuquerque_data.columns)}")
            return None, None, None
        
        # Pivot the data to wide format: block_group_fips as rows, years as columns
        print("Reshaping data to wide format...")
        income_wide = albuquerque_data.pivot(index='block_group_fips', columns='year', values='mean_income')
        population_wide = albuquerque_data.pivot(index='block_group_fips', columns='year', values='population')
        
        # Sort by year to ensure chronological order
        income_wide = income_wide.reindex(sorted(income_wide.columns), axis=1)
        population_wide = population_wide.reindex(sorted(population_wide.columns), axis=1)
        
        print(f"Reshaped data: {income_wide.shape[0]} block groups × {income_wide.shape[1]} years")
        print(f"Year range: {list(income_wide.columns)}")
        
        # Handle missing values
        print("Handling missing values...")
        n_missing_before = income_wide.isnull().sum().sum()
        if n_missing_before > 0:
            print(f"Found {n_missing_before} missing values, applying forward-fill...")
            
            # Forward-fill missing values within each block group
            for i in range(income_wide.shape[0]):
                income_wide.iloc[i, :] = income_wide.iloc[i, :].fillna(method='ffill').fillna(method='bfill')
                population_wide.iloc[i, :] = population_wide.iloc[i, :].fillna(method='ffill').fillna(method='bfill')
            
            n_missing_after = income_wide.isnull().sum().sum()
            print(f"After forward-fill: {n_missing_after} missing values remaining")
        
        # Calculate growth rates
        print("Calculating growth rates...")
        growth_rates_data = []
        population_weights = []
        
        for i in range(income_wide.shape[0]):
            # Get income and population for this block group
            incomes = income_wide.iloc[i, :].values
            populations = population_wide.iloc[i, :].values
            
            # Calculate log growth rates
            log_incomes = np.log(incomes)
            growth_rates = np.diff(log_incomes)
            
            # Only include if we have valid growth rates
            if np.all(np.isfinite(growth_rates)) and len(growth_rates) > 0:
                growth_rates_data.append(growth_rates)
                # Use average population as weight
                population_weights.append(np.mean(populations))
        
        years = list(income_wide.columns[1:])  # Growth rates start from second year
        
        print(f"✓ Successfully calculated growth rates:")
        print(f"  Block groups: {len(growth_rates_data)}")
        print(f"  Timesteps: {len(years)}")
        print(f"  Growth rate range: {np.min([np.min(gr) for gr in growth_rates_data]):.4f} to {np.max([np.max(gr) for gr in growth_rates_data]):.4f}")
        
        return growth_rates_data, population_weights, years
        
    else:
        print("✗ Data is not in expected DataFrame format")
        return None, None, None

def estimate_uncertainties_from_growth_rates(growth_rates_data, method='empirical'):
    """
    Estimate uncertainties for growth rates using various methods.
    This is only used as a fallback when precomputed uncertainties are not available.
    
    Parameters:
    -----------
    growth_rates_data : list
        List of growth rate arrays for each block group
    method : str
        Method to estimate uncertainties: 'empirical', 'rolling', or 'constant'
        
    Returns:
    --------
    list
        List of uncertainty arrays for each block group
    """
    print(f"\nEstimating uncertainties using '{method}' method (fallback)...")
    
    uncertainties_data = []
    
    if method == 'empirical':
        # Use empirical standard deviation across block groups at each timestep
        for i, agent_growth_rates in enumerate(growth_rates_data):
            n_timesteps = len(agent_growth_rates)
            uncertainties = np.zeros(n_timesteps)
            
            for t in range(n_timesteps):
                # Collect growth rates across all block groups at this timestep
                timestep_rates = []
                for other_rates in growth_rates_data:
                    if t < len(other_rates):
                        timestep_rates.append(other_rates[t])
                
                if len(timestep_rates) > 1:
                    # Use standard deviation across block groups as uncertainty
                    uncertainties[t] = np.std(timestep_rates)
                else:
                    # Fallback to small constant uncertainty
                    uncertainties[t] = 0.01
            
            uncertainties_data.append(uncertainties)
            
    elif method == 'rolling':
        # Use rolling standard deviation within each block group
        for agent_growth_rates in growth_rates_data:
            if len(agent_growth_rates) >= 3:
                # Calculate rolling standard deviation with window size 3
                uncertainties = np.zeros_like(agent_growth_rates)
                for t in range(len(agent_growth_rates)):
                    start_idx = max(0, t - 1)
                    end_idx = min(len(agent_growth_rates), t + 2)
                    window_rates = agent_growth_rates[start_idx:end_idx]
                    if len(window_rates) > 1:
                        uncertainties[t] = np.std(window_rates)
                    else:
                        uncertainties[t] = 0.01
            else:
                # Use small constant uncertainty for short series
                uncertainties = np.full_like(agent_growth_rates, 0.01)
            
            uncertainties_data.append(uncertainties)
            
    elif method == 'constant':
        # Use constant uncertainty based on overall variance
        overall_std = np.std([rate for rates in growth_rates_data for rate in rates])
        constant_uncertainty = max(overall_std * 0.1, 0.005)  # 10% of overall std, min 0.5%
        
        for agent_growth_rates in growth_rates_data:
            uncertainties = np.full_like(agent_growth_rates, constant_uncertainty)
            uncertainties_data.append(uncertainties)
    
    print(f"✓ Estimated uncertainties (fallback):")
    print(f"  Mean uncertainty: {np.mean([np.mean(u) for u in uncertainties_data]):.4f}")
    print(f"  Uncertainty range: {np.min([np.min(u) for u in uncertainties_data]):.4f} to {np.max([np.max(u) for u in uncertainties_data]):.4f}")
    
    return uncertainties_data

def main():
    """
    Main function to examine Albuquerque data and extract growth rates.
    """
    print("=" * 60)
    print("EXAMINING ALBUQUERQUE CBSA ACS DATA")
    print("=" * 60)
    
    # Load Albuquerque data
    albuquerque_data = load_albuquerque_data()
    if albuquerque_data is None:
        return
    
    # Examine data structure
    examine_data_structure(albuquerque_data)
    
    # Try to load precomputed data from urbandata
    urbandata_processed = load_urbandata_processed_data()
    
    # Extract growth rates (trying precomputed first, then falling back to calculation)
    growth_rates_data, population_weights, years, uncertainties = extract_growth_rates_from_albuquerque(
        albuquerque_data, urbandata_processed
    )
    
    if growth_rates_data is None:
        return
    
    # If we don't have precomputed uncertainties, estimate them
    if uncertainties is None:
        print("\n" + "=" * 50)
        print("ESTIMATING UNCERTAINTIES (FALLBACK)")
        print("=" * 50)
        
        uncertainty_methods = ['empirical', 'rolling', 'constant']
        all_uncertainties = {}
        
        for method in uncertainty_methods:
            uncertainties = estimate_uncertainties_from_growth_rates(growth_rates_data, method)
            all_uncertainties[method] = uncertainties
    else:
        print("✓ Using precomputed uncertainties from urbandata")
        all_uncertainties = {'precomputed': uncertainties}
    
    # Save extracted data for testing
    output_data = {
        'growth_rates_data': growth_rates_data,
        'population_weights': population_weights,
        'years': years,
        'uncertainties': all_uncertainties,
        'source': 'Albuquerque CBSA ACS data (with urbandata processing)'
    }
    
    output_file = "albuquerque_extracted_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"\n✓ Extracted data saved to {output_file}")
    print(f"Data summary:")
    print(f"  Block groups: {len(growth_rates_data)}")
    print(f"  Years: {years[0]} to {years[-1]} ({len(years)} timesteps)")
    print(f"  Uncertainty methods: {list(all_uncertainties.keys())}")
    
    print(f"\nNext steps:")
    print(f"1. Use this data to test vinference functions")
    print(f"2. Compare different uncertainty estimation methods")
    print(f"3. Validate results against known Albuquerque economic patterns")

if __name__ == "__main__":
    main() 