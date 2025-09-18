#!/usr/bin/env python3
"""
Data preparation pipeline for ACS economic indicators with ZIP code matching.
This script orchestrates the complete data preparation process:
1. Download ACS data from Census API
2. Match block groups to ZIP codes
3. Process data with growth rates and uncertainties
"""

import os
import sys
import subprocess
import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from census import Census

# --- ACS Data Retrieval Configuration ---
CENSUS_API_KEY = "35d314060d56f894db2f7621b0e5e5f7eca9af27"
if not CENSUS_API_KEY:
    raise ValueError("CENSUS_API_KEY environment variable not set. Please get a key from https://api.census.gov/data/key_signup.html")

YEARS = list(range(2014, 2024))
ACS_DATASET = 'acs5'

# Income bins and their midpoints for mean income calculation
INCOME_BINS = {
    'B19001_002E': 5000,
    'B19001_003E': 12500,
    'B19001_004E': 17500,
    'B19001_005E': 22500,
    'B19001_006E': 27500,
    'B19001_007E': 32500,
    'B19001_008E': 37500,
    'B19001_009E': 42500,
    'B19001_010E': 47500,
    'B19001_011E': 55000,
    'B19001_012E': 67500,
    'B19001_013E': 87500,
    'B19001_014E': 112500,
    'B19001_015E': 137500,
    'B19001_016E': 175000,
    'B19001_017E': 300000,  # Midpoint of $200,000 - $400,000 (user assumption)
}

# Variables to retrieve from ACS
ACS_VARIABLES = [
    'NAME',
    'B19001_001E',  # Total households
    'B19001_001M',  # MOE for total households
    'B25010_001E',  # Average household size
    'B25010_001M',  # MOE for average household size
] + list(INCOME_BINS.keys())

# Add MOE variables for income distribution
INCOME_MOE_VARIABLES = [
    'B19001_002M',  # MOE for <$10,000
    'B19001_003M',  # MOE for $10,000-$14,999
    'B19001_004M',  # MOE for $15,000-$19,999
    'B19001_005M',  # MOE for $20,000-$24,999
    'B19001_006M',  # MOE for $25,000-$29,999
    'B19001_007M',  # MOE for $30,000-$34,999
    'B19001_008M',  # MOE for $35,000-$39,999
    'B19001_009M',  # MOE for $40,000-$44,999
    'B19001_010M',  # MOE for $45,000-$49,999
    'B19001_011M',  # MOE for $50,000-$59,999
    'B19001_012M',  # MOE for $60,000-$74,999
    'B19001_013M',  # MOE for $75,000-$99,999
    'B19001_014M',  # MOE for $100,000-$124,999
    'B19001_015M',  # MOE for $125,000-$149,999
    'B19001_016M',  # MOE for $150,000-$199,999
    'B19001_017M',  # MOE for $200,000+
]

# Combine all variables
ACS_VARIABLES.extend(INCOME_MOE_VARIABLES)

def load_msa_data(filename="msa_fips_data.pkl"):
    """Loads the CBSA to FIPS codes mapping from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_acs_data(census_client, year, state_fips, county_fips):
    """Fetches ACS data for all block groups in a given county and year."""
    try:
        return census_client.acs5.get(
            ACS_VARIABLES,
            {'for': 'block group:*', 'in': f'state:{state_fips} county:{county_fips}'},
            year=year
        )
    except Exception as e:
        print(f"Could not retrieve data for state {state_fips}, county {county_fips}, year {year}: {e}")
        return None

def calculate_mean_income(row):
    """Calculates the mean household income from income distribution data."""
    total_households = row['B19001_001E']
    if total_households == 0:
        return 0
    
    weighted_income_sum = sum(row[bin_var] * midpoint for bin_var, midpoint in INCOME_BINS.items())
    return weighted_income_sum / total_households

def calculate_income_moe(row):
    """Calculates the margin of error for mean income using propagation of errors."""
    total_households = row['B19001_001E']
    if total_households == 0:
        return 0
    
    # Calculate weighted sum of squared MOEs
    weighted_moe_sum = 0
    for bin_var, midpoint in INCOME_BINS.items():
        moe_var = bin_var.replace('E', 'M')  # Replace estimate with MOE
        if moe_var in row:
            moe = row[moe_var]
            if not pd.isna(moe):
                weighted_moe_sum += (midpoint * moe) ** 2
    
    # MOE for mean = sqrt(sum of squared weighted MOEs) / total_households
    mean_moe = np.sqrt(weighted_moe_sum) / total_households
    return mean_moe

def calculate_population_moe(row):
    """Calculates the margin of error for population using propagation of errors."""
    households = row['B19001_001E']
    household_moe = row['B19001_001M']
    avg_size = row['B25010_001E']
    size_moe = row['B25010_001M']
    
    if pd.isna(households) or pd.isna(avg_size) or pd.isna(household_moe) or pd.isna(size_moe):
        return 0
    
    # Population MOE = sqrt((household_moe * avg_size)^2 + (households * size_moe)^2)
    pop_moe = np.sqrt((household_moe * avg_size) ** 2 + (households * size_moe) ** 2)
    return pop_moe

def calculate_income_cv(row):
    """Calculates the coefficient of variation for income estimates."""
    mean_income = row['mean_income']
    income_moe = row['income_moe']
    
    if mean_income == 0 or pd.isna(income_moe):
        return np.nan
    
    # CV = (MOE / 1.645) / mean_income * 100
    # 1.645 is the 90% confidence level z-score
    standard_error = income_moe / 1.645
    cv = (standard_error / mean_income) * 100
    return cv

def process_cbsa_data(cbsa_name, fips_codes, census_client):
    """Processes ACS data for a single CBSA over all specified years."""
    all_cbsa_data = []
    print(f"Processing CBSA: {cbsa_name}")
    
    # Track filtering statistics
    total_observations = 0
    filtered_observations = 0

    for year in YEARS:
        print(f"  Year: {year}")
        for state_fips, county_fips in fips_codes:
            raw_data = get_acs_data(census_client, year, state_fips, county_fips)
            if not raw_data:
                continue

            df = pd.DataFrame(raw_data)
            
            # Convert numeric columns to numeric types, coercing errors
            for col in ACS_VARIABLES:
                if col != 'NAME':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()

            if df.empty:
                continue

            # Apply data quality filters to remove problematic observations
            initial_count = len(df)
            total_observations += initial_count
            print(f"    Before filtering: {initial_count} block groups")
            
            # Filter out negative or zero household counts
            df = df[df['B19001_001E'] > 0]
            household_filtered = initial_count - len(df)
            
            # Filter out negative or zero household size
            df = df[df['B25010_001E'] > 0]
            size_filtered = initial_count - household_filtered - len(df)
            
            # Filter out extremely large household sizes (likely data errors)
            df = df[df['B25010_001E'] <= 50]  # Reasonable upper bound
            large_size_filtered = initial_count - household_filtered - size_filtered - len(df)
            
            # Filter out negative income values (though this should be redundant)
            df = df[df['B19001_001E'] > 0]  # Household count should be non-negative
            
            final_count = len(df)
            filtered_observations += (initial_count - final_count)
            
            print(f"    After filtering: {final_count} block groups")
            print(f"    Filtered out: {initial_count - final_count} observations")
            if household_filtered > 0:
                print(f"      - Invalid household count: {household_filtered}")
            if size_filtered > 0:
                print(f"      - Invalid household size: {size_filtered}")
            if large_size_filtered > 0:
                print(f"      - Extremely large household size: {large_size_filtered}")
            
            if df.empty:
                print(f"    No valid data after filtering for {cbsa_name}, {year}")
                continue

            df['mean_income'] = df.apply(calculate_mean_income, axis=1)
            df['population'] = df['B19001_001E'] * df['B25010_001E']
            
            # Calculate error measures
            df['income_moe'] = df.apply(calculate_income_moe, axis=1)
            df['population_moe'] = df.apply(calculate_population_moe, axis=1)
            
            # Calculate coefficient of variation for income
            df['income_cv'] = df.apply(calculate_income_cv, axis=1)
            
            df['year'] = year
            df['block_group_fips'] = df['state'] + df['county'] + df['tract'] + df['block group']

            all_cbsa_data.append(df[['year', 'block_group_fips', 'mean_income', 'income_moe', 'income_cv', 'population', 'population_moe']])
    
    if not all_cbsa_data:
        return None

    print(f"  Total observations processed: {total_observations:,}")
    print(f"  Total observations filtered: {filtered_observations:,}")
    print(f"  Filtering rate: {filtered_observations/total_observations*100:.2f}%")

    return pd.concat(all_cbsa_data, ignore_index=True)

def run_acs_data_retrieval():
    """Run ACS data retrieval with real-time progress tracking."""
    print(f"\n{'='*60}")
    print("STEP: ACS Data Retrieval from Census API")
    print(f"{'='*60}")
    
    # Check if the data already exists to avoid re-downloading
    acs_data_file = "data_retrieval/cbsa_acs_data.pkl"
    if os.path.exists(acs_data_file):
        print("ACS data file 'cbsa_acs_data.pkl' already exists.")
        print("Deleting existing file to re-download with error measures...")
        os.remove(acs_data_file)
        print("Existing file deleted. Proceeding with download...")

    try:
        c = Census(CENSUS_API_KEY)
        msa_fips_data = load_msa_data("data_retrieval/msa_fips_data.pkl")
        
        print(f"Starting data retrieval for {len(msa_fips_data)} CBSAs...")
        
        all_cbsa_acs_data = {}
        for i, (cbsa, fips) in enumerate(msa_fips_data.items(), 1):
            print(f"\n[{i}/{len(msa_fips_data)}] Processing {cbsa}...")
            cbsa_df = process_cbsa_data(cbsa, fips, c)
            if cbsa_df is not None:
                all_cbsa_acs_data[cbsa] = cbsa_df
                print(f"✅ {cbsa} completed successfully")
            else:
                print(f"❌ {cbsa} failed to process")

        # Save the data
        with open(acs_data_file, 'wb') as f:
            pickle.dump(all_cbsa_acs_data, f)

        print(f"\n✅ Successfully retrieved and processed ACS data for {len(all_cbsa_acs_data)} CBSAs.")
        print(f"Data saved to {acs_data_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error during ACS data retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_script(script_path, description, cwd=None):
    """Run a Python script and handle any errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script_path}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print output
        if result.stdout:
            print("Output:")
            print(result.stdout)
        
        print(f"✅ {description} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error running {description}: {e}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists and print its status."""
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"✅ {description}: {file_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def main():
    """Main data preparation pipeline."""
    print("🚀 Starting ACS Data Preparation Pipeline")
    print(f"Current working directory: {os.getcwd()}")
    
    # Define paths
    base_dir = Path(__file__).parent
    data_retrieval_dir = base_dir / "data_retrieval"
    zip_matching_dir = base_dir / "zip_matching"
    
    # Step 1: ACS Data Retrieval (Integrated)
    success = run_acs_data_retrieval()
    
    if not success:
        print("❌ ACS data retrieval failed. Stopping pipeline.")
        return False
    
    # Check if ACS data was created
    acs_data_file = data_retrieval_dir / "cbsa_acs_data.pkl"
    if not check_file_exists(acs_data_file, "ACS Data File"):
        print("❌ ACS data file not created. Stopping pipeline.")
        return False
    
    # Step 2: ZIP Code Matching
    zip_script = zip_matching_dir / "match_zips.py"
    if not zip_script.exists():
        print(f"❌ ZIP matching script not found: {zip_script}")
        return False
    
    success = run_script(
        zip_script,
        "ZIP Code Matching for Block Groups",
        cwd=zip_matching_dir
    )
    
    if not success:
        print("❌ ZIP matching failed. Stopping pipeline.")
        return False
    
    # Check if ZIP-matched data was created
    zip_data_file = zip_matching_dir / "blockgroups_with_zips_temporal.pkl"
    if not check_file_exists(zip_data_file, "ZIP-Matched Data File"):
        print("❌ ZIP-matched data file not created. Stopping pipeline.")
        return False
    
    # Step 3: Data Processing with Uncertainties
    process_script = base_dir / "process_acs_data.py"
    if not process_script.exists():
        print(f"❌ Data processing script not found: {process_script}")
        return False
    
    success = run_script(
        process_script,
        "Data Processing with Growth Rates and Uncertainties",
        cwd=base_dir
    )
    
    if not success:
        print("❌ Data processing failed. Stopping pipeline.")
        return False
    
    # Check final output
    final_data_file = base_dir / "processed_acs_data.pkl"
    if not check_file_exists(final_data_file, "Final Processed Data File"):
        print("❌ Final processed data file not created.")
        return False
    
    # Pipeline complete
    print(f"\n{'='*60}")
    print("🎉 DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\nOutput files created:")
    print(f"1. ACS Raw Data: {acs_data_file}")
    print(f"2. ZIP-Matched Data: {zip_data_file}")
    print(f"3. Final Processed Data: {final_data_file}")
    print("\nThe processed data now includes:")
    print("- Log growth rates with uncertainties")
    print("- P-values (fraction of positive growth) with uncertainties")
    print("- ZIP code mappings for each block group")
    print("- Population-weighted statistics")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Pipeline failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Pipeline completed successfully!") 