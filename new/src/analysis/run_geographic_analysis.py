#!/usr/bin/env python3
"""
Executes the multi-level environmental parameter inference pipeline.

This script orchestrates the following workflow:
1.  Loads pre-processed ACS data for multiple MSAs.
2.  Calculates log-income growth rates for each block group.
3.  Constructs an analysis pipeline with tasks for different geographic
    aggregation levels (MSA, county, ZIP, tract).
4.  (Future) Executes the inference models for each task in the pipeline.
"""

import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- Setup Project Paths ---
# This ensures that the script can find the custom modules (e.g., in inference)
try:
    project_root = Path(__file__).resolve().parent.parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from inference.vinference import bayesian_p_estimation
    from inference.ssm_model import fit_l_and_x_hierarchical
except ImportError:
    print("Error: Could not import project modules.")
    print("Please ensure the script is run from a location where 'src' is accessible.")
    sys.exit(1)


def preprocess_data(msa_data):
    """
    Prepares the raw MSA data for inference by calculating growth rates.

    Args:
        msa_data (dict): Dictionary of MSA names to DataFrames.

    Returns:
        dict: The same dictionary with modified DataFrames that include
              'tract_fips' and 'growth_rate' columns.
    """
    print("Phase 1: Pre-processing data...")
    for msa_name, df in msa_data.items():
        print(f"  - Processing {msa_name}...")
        # Ensure block_group_fips is a string for manipulation
        df['block_group_fips'] = df['block_group_fips'].astype(str)

        # Calculate Tract FIPS from block group FIPS
        df['tract_fips'] = df['block_group_fips'].str[:11]

        # Calculate log-income growth rates
        # Replace 0 or negative incomes with NaN before taking log
        df['mean_income_positive'] = df['mean_income'].where(df['mean_income'] > 0)
        df['log_income'] = np.log(df['mean_income_positive'])

        # Sort values to ensure correct 'diff' calculation
        df.sort_values(by=['block_group_fips', 'year'], inplace=True)
        # Group by block group and calculate the difference in log_income year-over-year
        df['growth_rate'] = df.groupby('block_group_fips')['log_income'].diff()
        
        # Clean up temporary columns
        df.drop(columns=['mean_income_positive', 'log_income'], inplace=True)
    
    print("✓ Pre-processing complete.\n")
    return msa_data


def construct_pipeline(msa_data):
    """
    Builds a list of all analysis tasks to be run.

    Each task is a unique combination of an MSA and a geographic aggregation level.

    Args:
        msa_data (dict): The pre-processed dictionary of MSA DataFrames.

    Returns:
        list: A list of dictionaries, where each dictionary represents an
              inference task.
    """
    print("Phase 2: Constructing analysis pipeline...")
    analysis_pipeline = []
    
    for msa_name, df in msa_data.items():
        # MSA Level Task
        analysis_pipeline.append({
            'msa': msa_name,
            'level': 'msa',
            'group_id': msa_name
        })

        # County Level Tasks
        for county_fips in df['county_fips'].unique():
            analysis_pipeline.append({
                'msa': msa_name,
                'level': 'county',
                'group_id': county_fips
            })

        # ZIP Code Level Tasks
        for zip_code in df['closest_zip'].unique():
            analysis_pipeline.append({
                'msa': msa_name,
                'level': 'zip',
                'group_id': zip_code
            })
            
        # Tract Level Tasks
        for tract_fips in df['tract_fips'].unique():
            analysis_pipeline.append({
                'msa': msa_name,
                'level': 'tract',
                'group_id': tract_fips
            })

    print(f"✓ Pipeline constructed with {len(analysis_pipeline)} tasks.\n")
    return analysis_pipeline


def execute_pipeline(msa_data, pipeline):
    """
    Runs the inference models for each task in the pipeline.

    Args:
        msa_data (dict): The pre-processed dictionary of MSA DataFrames.
        pipeline (list): The list of analysis tasks.

    Returns:
        dict: A dictionary containing the inference results for each task.
    """
    print("Phase 3: Executing inference pipeline...")
    results = {}
    
    # For progress tracking
    from tqdm import tqdm

    for task in tqdm(pipeline, desc="Processing tasks"):
        msa_name = task['msa']
        level = task['level']
        group_id = task['group_id']
        task_key = f"{msa_name}_{level}_{group_id}"

        # 1. Filter data for the current group
        df = msa_data[msa_name]
        if level == 'msa':
            group_df = df
        elif level == 'county':
            group_df = df[df['county_fips'] == group_id]
        elif level == 'zip':
            group_df = df[df['closest_zip'] == group_id]
        elif level == 'tract':
            group_df = df[df['tract_fips'] == group_id]

        # 2. Prepare model inputs
        # Pivot to get agents (block groups) as rows and years as columns
        pivot_df = group_df.pivot_table(
            index='block_group_fips', columns='year', values='growth_rate'
        )
        # The first column of years will be NaN due to diff(), so we drop it.
        if not pivot_df.empty:
            pivot_df = pivot_df.iloc[:, 1:]

        # Skip if there are not enough agents for a robust analysis
        if len(pivot_df) < 10:
            continue

        growth_rates_data = [row.values for _, row in pivot_df.iterrows()]
        
        # Get population weights aligned with the agents in the pivot table
        latest_year = group_df['year'].max()
        pop_df = group_df[group_df['year'] == latest_year].set_index('block_group_fips')
        aligned_pop = pop_df.reindex(pivot_df.index)
        population_weights = aligned_pop['population'].fillna(1).tolist()

        try:
            # 3. Estimate p_t series
            p_t_series, _, valid_timesteps = bayesian_p_estimation(
                growth_rates_data, population_weights, min_agents_per_timestep=1, verbose=False
            )

            if len(p_t_series) == 0:
                continue

            growth_years = pivot_df.columns.tolist()
            valid_years = [growth_years[i] for i in valid_timesteps]

            # --- NEW: Print summary line with data availability ---
            # Calculate the number of valid agents for each of the timesteps that had a p_t value
            valid_counts_per_year = pivot_df.iloc[:, valid_timesteps].notna().sum().tolist()

            p_list_str = ", ".join([f"{p:.2f}" for p in p_t_series])
            counts_list_str = ", ".join(map(str, valid_counts_per_year))
            
            # Build hierarchy string for reporting
            if level == 'msa':
                hierarchy_str = f"MSA: {msa_name}"
            elif level == 'county':
                hierarchy_str = f"MSA: {msa_name}, County: {group_id}"
            elif level == 'zip':
                hierarchy_str = f"MSA: {msa_name}, ZIP: {group_id}"
            elif level == 'tract':
                county_fips = str(group_id)[:5]
                hierarchy_str = f"MSA: {msa_name}, County: {county_fips}, Tract: {group_id}"
            
            print(f"{hierarchy_str:<100} | Valid Agents/Year: [{counts_list_str:<25}] | p_t: [{p_list_str}]")
            # --- END NEW ---

            # 4. Estimate l_t series, timestep by timestep
            l_t_series = []
            for i, p_hat_t in enumerate(p_t_series):
                year_t = valid_years[i]
                
                # Get all non-NaN growth rates for the current year
                y_values_t = pivot_df[year_t].dropna().values
                
                # Get population weights for agents with data in the current year
                agents_with_data_t = pivot_df[year_t].dropna().index
                population_weights_t = aligned_pop.loc[agents_with_data_t]['population'].fillna(1).tolist()

                # Hierarchical model needs a minimum number of data points
                if len(y_values_t) < 10:
                    l_t_series.append(np.nan)
                    continue
                
                try:
                    idata = fit_l_and_x_hierarchical(
                        y_values_t, p_hat_t, population_weights=population_weights_t
                    )
                    l_mean = idata.posterior["l"].mean().item()
                    l_t_series.append(l_mean)
                except Exception as e:
                    print(f"\n---! ERROR running hierarchical model for task {task_key}, year {year_t} !---")
                    import traceback
                    traceback.print_exc()
                    print("---------------------------------------------------------------------------------")
                    l_t_series.append(np.nan)

            # --- NEW: Print l_t summary line ---
            l_list_str = ", ".join([f"{l:.2f}" for l in l_t_series])
            aligned_hierarchy_str = " " * 100  # Create padding to align with the line above
            print(f"{aligned_hierarchy_str} | {'':<25}   | l_t: [{l_list_str}]")
            # --- END NEW ---

            # 5. Store results
            results[task_key] = {
                'p_t_series': p_t_series,
                'l_t_series': np.array(l_t_series),
                'years': valid_years
            }
        except Exception as e:
            # If bayesian_p_estimation fails, skip the task
            print(f"\n---! ERROR in p_t estimation for task {task_key} !---")
            import traceback
            traceback.print_exc()
            print("---------------------------------------------------------------------------------")
            continue
            
    print(f"\n✓ Pipeline execution complete. {len(results)} tasks successfully processed.")
    return results


def main():
    """
    Main function to run the entire analysis pipeline.
    """
    # --- Load All Data ---
    # The new data source is a single CSV file. We load it and then reconstruct
    # the dictionary-of-DataFrames format that the rest of the script expects.
    data_path = project_root / "data" / "processed" / "blockgroups_with_zips_temporal.csv"
    print(f"Loading data from {data_path}...")
    try:
        # Use the legacy path if the new one doesn't exist, for backward compatibility
        legacy_path = project_root / "legacy" / "urbandata" / "data" / "zip_matching" / "blockgroups_with_zips_temporal.csv"
        if not data_path.exists() and legacy_path.exists():
            print(f"  > Note: Using legacy data file at {legacy_path}")
            data_path = legacy_path

        combined_df = pd.read_csv(data_path)
        
        # Reconstruct the dictionary format: {msa_name: msa_dataframe}
        all_msa_data = {
            msa_name: msa_df.drop(columns='msa_name').reset_index(drop=True)
            for msa_name, msa_df in combined_df.groupby('msa_name')
        }
        print(f"✓ Loaded and reshaped data for {len(all_msa_data)} MSAs.\n")

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path} or {legacy_path}")
        return

    # --- Set target MSAs for processing ---
    # To run for all, set target_msas = None
    target_msas = ['Atlanta-Sandy Springs-Roswell, GA']
    
    if target_msas:
        msa_to_process = {k: v for k, v in all_msa_data.items() if k in target_msas}
        print(f"--- Running analysis for {len(msa_to_process)} target MSA(s) ---\n")
    else:
        msa_to_process = all_msa_data
        print("--- Running analysis for all MSAs ---\n")

    idx = int(sys.argv[1])
    # --- Main Loop: Process and Save One MSA at a Time ---
    for msa_name, msa_df in [list(msa_to_process.items())[idx]]:
        print(f"\n{'='*20} Processing MSA: {msa_name} {'='*20}")
        current_msa_data = {msa_name: msa_df}

        # Phase 1: Pre-processing
        msa_data_processed = preprocess_data(current_msa_data)

        # Phase 2: Pipeline Construction
        pipeline = construct_pipeline(msa_data_processed)

        # Phase 3: Execution Engine
        results = execute_pipeline(msa_data_processed, pipeline)

        # Save Results for the current MSA
        if results:
            # Sanitize MSA name for use in filename
            sanitized_name = msa_name.replace(' ', '_').replace(',', '').replace('/', '_')
            results_path = project_root / "data" / "processed" / f"{sanitized_name}_results.pkl"
            print(f"\nSaving results for {msa_name} to {results_path}...")
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            print("✓ Results saved successfully.")

    print(f"\n{'='*20} Full Pipeline Finished {'='*20}")


if __name__ == "__main__":
    main() 
