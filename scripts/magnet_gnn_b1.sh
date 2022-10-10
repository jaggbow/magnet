#!/bin/bash

#SBATCH --job-name=magnet_gnn_b1
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1      
#SBATCH --mem=60G                                     
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/magnet_gnn_b1-slurm-%j.out  

# 1. Load the required modules
module --quiet load anaconda/3
conda activate dedalus

python test_reg_b1.py