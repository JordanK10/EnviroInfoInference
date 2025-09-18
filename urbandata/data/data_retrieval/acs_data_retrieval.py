import os
import pickle
import pandas as pd
import numpy as np
from census import Census

# --- Configuration ---
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

def calculate_growth_rate_with_uncertainty(current_data, previous_data):
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
    
    # Calculate growth rate
    merged['growth_rate'] = (merged['mean_income_current'] - merged['mean_income_previous']) / merged['mean_income_previous']
    
    # Calculate uncertainty in growth rate using error propagation
    # For f(x,y) = (x-y)/y, the uncertainty is:
    # σ_f = sqrt((∂f/∂x * σ_x)² + (∂f/∂y * σ_y)²)
    # where ∂f/∂x = 1/y and ∂f/∂y = -x/y²
    
    # Convert MOE to standard error (MOE = 1.645 * SE for 90% confidence)
    merged['income_se_current'] = merged['income_moe_current'] / 1.645
    merged['income_se_previous'] = merged['income_moe_previous'] / 1.645
    
    # Partial derivatives
    # ∂f/∂x = 1/y_previous
    # ∂f/∂y = -x_current / (y_previous)²
    merged['partial_x'] = 1.0 / merged['mean_income_previous']
    merged['partial_y'] = -merged['mean_income_current'] / (merged['mean_income_previous'] ** 2)
    
    # Calculate growth rate uncertainty
    merged['growth_rate_moe'] = np.sqrt(
        (merged['partial_x'] * merged['income_moe_current']) ** 2 + 
        (merged['partial_y'] * merged['income_moe_previous']) ** 2
    )
    
    # Calculate coefficient of variation for growth rate
    merged['growth_rate_cv'] = (merged['growth_rate_moe'] / 1.645) / np.abs(merged['growth_rate']) * 100
    
    # Filter out infinite or NaN values
    merged = merged[np.isfinite(merged['growth_rate'])]
    
    return merged[['block_group_fips', 'growth_rate', 'growth_rate_moe', 'growth_rate_cv']]

def calculate_population_weighted_growth_rate_with_uncertainty(growth_data):
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
    weighted_variance = ((growth_data['population'] ** 2) * (growth_data['growth_rate_moe'] / 1.645) ** 2).sum()
    weighted_moe = np.sqrt(weighted_variance) / total_population * 1.645
    
    # Weighted coefficient of variation
    weighted_cv = (weighted_moe / 1.645) / np.abs(weighted_mean) * 100
    
    return {
        'weighted_mean': weighted_mean,
        'weighted_moe': weighted_moe,
        'weighted_cv': weighted_cv,
        'total_population': total_population
    }

def calculate_p_with_uncertainty(growth_data):
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


def main():
    """Main function to orchestrate the data retrieval and processing."""
    
    # Check if the data already exists to avoid re-downloading
    if os.path.exists('cbsa_acs_data.pkl'):
        print("ACS data file 'cbsa_acs_data.pkl' already exists.")
        print("Deleting existing file to re-download with error measures...")
        os.remove('cbsa_acs_data.pkl')
        print("Existing file deleted. Proceeding with download...")

    c = Census(CENSUS_API_KEY)
    msa_fips_data = load_msa_data()
    
    all_cbsa_acs_data = {}
    for cbsa, fips in msa_fips_data.items():
        cbsa_df = process_cbsa_data(cbsa, fips, c)
        if cbsa_df is not None:
            all_cbsa_acs_data[cbsa] = cbsa_df

    with open('cbsa_acs_data.pkl', 'wb') as f:
        pickle.dump(all_cbsa_acs_data, f)

    print("\nSuccessfully retrieved and processed ACS data for all CBSAs.")
    print("Data saved to cbsa_acs_data.pkl")


if __name__ == "__main__":
    main() 