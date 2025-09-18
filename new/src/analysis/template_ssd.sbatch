#!/bin/sh
#SBATCH --job-name=geographic_NUMM
#SBATCH --output=/home/andrewstier/EnviroInfoInference/new/src/analysis/geographic_NUMM.out
#SBATCH --error=/home/andrewstier/EnviroInfoInference/new/src/analysis/geographic_NUMM.err
#SBATCH --partition=ssd
#SBATCH --account=ssd
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G



module load python
source activate /project/bermanm/conda_envs/fmri_processing
module unload python

python3 /home/andrewstier/EnviroInfoInference/new/src/analysis/run_geographic_analysis.py NUMM
