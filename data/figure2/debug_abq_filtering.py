#!/usr/bin/env python3
"""
Debug Albuquerque Data Filtering
================================

This script examines the processed Albuquerque data to understand
why population and income data don't align.
"""

import pickle
import pandas as pd
import numpy as np
import os

def debug_albuquerque_filtering():
    """Debug the filtering process for Albuquerque data."""
    
    # Path to CBSA ACS data
    data_path = "figure1/cbsa_acs_data.pkl"
    
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        return
    
    # Load the data
    print("Loading CBSA ACS data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Find Albuquerque data
    albuquerque_data = None
    for cbsa_name, cbsa_data in data.items():
        if 'albuquerque' in cbsa_name.lower() and 'nm' in cbsa_name.lower():
            albuquerque_data = cbsa_data
            print(f"✓ Found Albuquerque, NM data: {cbsa_name}")
            break
    
    if albuquerque_data is None:
        print("✗ Could not find Albuquerque, NM data")
        return
    
    print(f"\nAlbuquerque data shape: {albuquerque_data.shape}")
    print(f"Columns: {list(albuquerque_data.columns)}")
    
    # Check data completeness
    print(f"\n=== DATA COMPLETENESS ANALYSIS ===")
    
    # Check for missing values in each column
    for col in albuquerque_data.columns:
        missing_count = albuquerque_data[col].isna().sum()
        total_count = len(albuquerque_data)
        print(f"{col}: {missing_count}/{total_count} missing ({missing_count/total_count*100:.1f}%)")
    
    # Analyze the processed data structure
    print(f"\n=== ANALYZING PROCESSED DATA ===")
    
    # Check for zero/negative values in processed data
    initial_count = len(albuquerque_data)
    print(f"Total observations: {initial_count}")
    
    # Check population data
    zero_population = (albuquerque_data['population'] == 0).sum()
    negative_population = (albuquerque_data['population'] < 0).sum()
    print(f"\nPopulation analysis:")
    print(f"  Zero population: {zero_population}/{initial_count} ({zero_population/initial_count*100:.1f}%)")
    print(f"  Negative population: {negative_population}/{initial_count} ({negative_population/initial_count*100:.1f}%)")
    print(f"  Population range: {albuquerque_data['population'].min():.0f} to {albuquerque_data['population'].max():.0f}")
    
    # Check income data
    zero_income = (albuquerque_data['mean_income'] == 0).sum()
    negative_income = (albuquerque_data['mean_income'] < 0).sum()
    print(f"\nIncome analysis:")
    print(f"  Zero income: {zero_income}/{initial_count} ({zero_income/initial_count*100:.1f}%)")
    print(f"  Negative income: {negative_income}/{initial_count} ({negative_income/initial_count*100:.1f}%)")
    print(f"  Income range: ${albuquerque_data['mean_income'].min():.0f} to ${albuquerque_data['mean_income'].max():.0f}")
    
    # Check data by year
    print(f"\n=== DATA BY YEAR ===")
    for year in sorted(albuquerque_data['year'].unique()):
        year_data = albuquerque_data[albuquerque_data['year'] == year]
        year_count = len(year_data)
        year_pop = year_data['population'].sum()
        year_income_avg = year_data['mean_income'].mean()
        print(f"  {year}: {year_count} block groups, total pop: {year_pop:,.0f}, avg income: ${year_income_avg:,.0f}")
    
    # Check unique block groups
    unique_block_groups = albuquerque_data['block_group_fips'].nunique()
    print(f"\nUnique block groups: {unique_block_groups}")
    
    # Check if the issue is in the pivot operation
    print(f"\n=== TESTING PIVOT OPERATION ===")
    
    # Try to pivot the data to see what happens
    try:
        # Pivot to wide format (block groups as rows, years as columns)
        resources = albuquerque_data.pivot(index='block_group_fips', columns='year', values='mean_income')
        print(f"Pivoted income data shape: {resources.shape}")
        print(f"Years: {list(resources.columns)}")
        
        # Check for missing values in pivoted data
        missing_income = resources.isna().sum().sum()
        total_income_cells = resources.size
        print(f"Income data missing: {missing_income}/{total_income_cells} ({missing_income/total_income_cells*100:.1f}%)")
        
        # Pivot population data
        population_wide = albuquerque_data.pivot(index='block_group_fips', columns='year', values='population')
        print(f"Pivoted population data shape: {population_wide.shape}")
        
        # Check for missing values in population data
        missing_pop = population_wide.isna().sum().sum()
        total_pop_cells = population_wide.size
        print(f"Population data missing: {missing_pop}/{total_pop_cells} ({missing_pop/total_pop_cells*100:.1f}%)")
        
        # Compare the two
        print(f"\nComparison:")
        print(f"  Income data: {resources.shape[0]} block groups × {resources.shape[1]} years")
        print(f"  Population data: {population_wide.shape[0]} block groups × {population_wide.shape[1]} years")
        
        if resources.shape[0] != population_wide.shape[0]:
            print(f"  ⚠️  Different number of block groups!")
            print(f"  Income block groups: {resources.shape[0]}")
            print(f"  Population block groups: {population_wide.shape[0]}")
            
            # Find the difference
            income_bgs = set(resources.index)
            pop_bgs = set(population_wide.index)
            only_in_income = income_bgs - pop_bgs
            only_in_pop = pop_bgs - income_bgs
            
            if only_in_income:
                print(f"  Block groups only in income data: {len(only_in_income)}")
            if only_in_pop:
                print(f"  Block groups only in population data: {len(only_in_pop)}")
        
    except Exception as e:
        print(f"Pivot operation failed: {e}")
    
    # Check what the actual data looks like
    print(f"\n=== ACTUAL DATA STRUCTURE ===")
    print(f"Data types:")
    for col in albuquerque_data.columns:
        print(f"  {col}: {albuquerque_data[col].dtype}")
    
    print(f"\nSample data (first 3 rows):")
    print(albuquerque_data.head(3).to_string())
    
    # Check if this is the processed data or raw data
    if 'mean_income' in albuquerque_data.columns and 'population' in albuquerque_data.columns:
        print(f"\nThis appears to be PROCESSED data (has mean_income and population columns)")
        print(f"Raw ACS variables may not be available for detailed filtering analysis")
    else:
        print(f"\nThis appears to be RAW ACS data")
        print(f"Can perform detailed filtering analysis")

if __name__ == "__main__":
    debug_albuquerque_filtering() 