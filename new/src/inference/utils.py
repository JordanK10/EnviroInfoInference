"""
Shared utilities for the EnviroInfoInference project.
"""

import os
import sys
from pathlib import Path

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def setup_project_paths():
    """Add src directory to Python path using absolute project root."""
    project_root = get_project_root()
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    return project_root, src_path

# Mathematical utilities
def logit(p):
    """Convert probability to logit."""
    import numpy as np
    return np.log(p) - np.log(1 - p)

def invlogit(x):
    """Convert logit to probability."""
    import numpy as np
    return 1 / (1 + np.exp(-x))

def clip_probability(p, min_val=1e-5, max_val=1-1e-5):
    """Clip probability to valid range."""
    import numpy as np
    return np.clip(p, min_val, max_val)
