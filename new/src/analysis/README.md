# Geographic Analysis Pipeline

This directory contains the scripts for applying the project's inference models to real-world ACS data.

## `run_geographic_analysis.py`

This is the main script for orchestrating the end-to-end analysis pipeline. It is designed to be the primary entry point for large-scale inference runs.

### Workflow

The script executes a multi-phase workflow:

1.  **Phase 1: Data Loading & Pre-processing**
    *   Loads the master data file: `data/processed/blockgroups_with_zips_temporal.pkl`.
    *   For each MSA, it calculates the necessary log-income `growth_rate` for every block group, which is the primary input for the models.

2.  **Phase 2: Pipeline Construction**
    *   Builds a comprehensive list of every "task" to be run.
    *   A task is a unique combination of a metropolitan area and a geographic aggregation level (MSA, county, ZIP code, or census tract).

3.  **Phase 3: Execution Engine**
    *   Iterates through every task in the pipeline.
    *   For each task (e.g., Fulton County in the Atlanta MSA):
        1.  **Infer `p_t`**: Calls `bayesian_p_estimation` to calculate the environmental predictability time series for that specific geographic group.
        2.  **Infer `l_t`**: For each year, it calls the `fit_l_and_x_hierarchical` model to calculate the environmental payout.
    *   Prints a real-time summary of the `p_t` and `l_t` series for each task as it completes.

### Usage

To run the script, navigate to the project's root directory (`new/`) and execute:

```bash
python src/analysis/run_geographic_analysis.py
```

By default, the script is configured to run on a single test MSA ('Atlanta-Sandy Springs-Roswell, GA'). To run on all MSAs, you must comment out or remove the filtering block in the `main` function.

### Output

The script generates a single output file:

*   **`data/processed/geographic_analysis_results.pkl`**: A Python pickle file containing a dictionary.
    *   **Keys**: A unique string for each successfully processed task (e.g., `"Atlanta-Sandy Springs-Roswell, GA_county_13121"`).
    *   **Values**: A dictionary containing the inferred `p_t_series`, `l_t_series`, and the corresponding `years`.

### Parallelization for Cluster Computing

The script is designed to be easily parallelized, where each MSA is processed independently on a separate node of a computing cluster.

The ideal place to implement this logic is in the `main` function of `run_geographic_analysis.py`, immediately after the data is loaded. A special comment block, **`--- PARALLELIZATION HOOK ---`**, has been added to the code to mark this exact spot.

To parallelize the script, you would modify this section to read a command-line argument or an environment variable (e.g., `SLURM_ARRAY_TASK_ID` from a SLURM scheduler). This argument would specify which MSA the script instance is responsible for. The script would then filter the master `msa_data` dictionary down to just that single MSA before proceeding with the rest of the pipeline.

**Conceptual Example (inside `main` function):**

```python
# (Code to load the full msa_data dictionary)

# --- PARALLELIZATION HOOK ---
# Example logic for selecting one MSA based on a command-line argument
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--msa_name", type=str, required=True, help="Specific MSA to process")
args = parser.parse_args()

if args.msa_name in msa_data:
    msa_data = {args.msa_name: msa_data[args.msa_name]}
else:
    print(f"Error: MSA '{args.msa_name}' not found.")
    exit()
# --- END PARALLELIZATION HOOK ---

# (The rest of the script proceeds with the single-MSA dictionary)
``` 