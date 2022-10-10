#!/bin/bash

#SBATCH --job-name=mpnn_test_uniform
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1      
#SBATCH --mem=60G                                     
#SBATCH --time=4:30:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/mpnn_test_uniform-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate dedalus

python test_irr_uniform.py