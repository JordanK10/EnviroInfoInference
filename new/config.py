"""
Configuration settings for EnviroInfoInference project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VALIDATION_DATA_DIR = DATA_DIR / "validation"
FIGURES_DIR = PROJECT_ROOT / "figures"
LEGACY_DIR = PROJECT_ROOT / "legacy"

# Model parameters
DEFAULT_L_MIN = 1.5
DEFAULT_L_MAX = 3.0
DEFAULT_L_GRID_SIZE = 50
DEFAULT_PRIOR_MEAN = 0.6
DEFAULT_PRIOR_CONCENTRATION = 3.0

# Validation parameters
DEFAULT_N_AGENTS = 100
DEFAULT_N_TIMESTEPS = 50
DEFAULT_INITIAL_RESOURCES = 50000
DEFAULT_SEED = 42

# Analysis parameters
MIN_AGENTS_PER_TIMESTEP = 5
DEFAULT_POPULATION_WEIGHT = 1.0

# File paths
CBSA_DATA_FILE = RAW_DATA_DIR / "cbsa_acs_data.pkl"
ALBUQUERQUE_DATA_FILE = PROCESSED_DATA_DIR / "albuquerque_extracted_data.pkl"
DUMMY_DATA_FILE = VALIDATION_DATA_DIR / "dummy_data_kelly_betting_static_x.pkl"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VALIDATION_DATA_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
