
import os
import sys
import numpy
import time
import glob
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

data_path = project_root / "data" / "processed" / "blockgroups_with_zips_temporal.pkl"
print(f"Loading data from {data_path}...")
try:
    with open(data_path, 'rb') as f:
        all_msa_data = pickle.load(f)
    print(f"✓ Loaded data for {len(all_msa_data)} MSAs.\n")
except FileNotFoundError:
    print(f"Error: Data file not found at {data_path}")


with open('/home/andrewstier/EnviroInfoInference/new/src/analysis/template_ssd.sbatch','r') as f:
	template = f.read()


for i in range(len(min_max)):
	new = template.replace('NUMM',str(i))
	sbatch_file = '/home/andrewstier/EnviroInfoInference/new/src/analysis/tmp/run_city_%d.sbatch' % i
	with open(sbatch_file,'w') as f:
		f.write(new)

	os.system('sbatch %s' % sbatch_file)

