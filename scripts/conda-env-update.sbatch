#!/bin/bash

#SBATCH --job-name=condaEnvUpdate
#SBATCH --nodes=1
#SBATCH --partition=gpu_inter
#SBATCH --time=1:00:00
#SBATCH --output=logsenv/slurm-%j.out
#SBATCH --error=logsenv/slurm-%j.err

# Load the conda module
export PATH=/opt/conda/bin:$PATH

# Activate the environment
source activate nlp
pip install wandb 

# You can also trigger "conda install ..." commands
echo "Finished !"