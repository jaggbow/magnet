#!/bin/bash

#SBATCH --job-name=magnet_gnn_2d_b2_64_regular
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:2      
#SBATCH --mem=60G                                     
#SBATCH --time=9:00:00
#SBATCH --array=1-5  
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/magnet_gnn_2d_b2_64_regular-slurm-%A_%a.out  

param_store=scripts/seeds.txt
seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
# 1. Load the required modules
module --quiet load anaconda/3
conda activate dedalus

python run.py \
seed=$seed \
model=magnet_gnn \
name=magnet_gnn \
datamodule=h5_datamodule_implicit_gnn_2d \
datamodule.train_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B2/burgers_train_B2_64.h5' \
datamodule.val_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B2/burgers_test_B2_64.h5' \
datamodule.test_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B2/burgers_test_B2_64.h5' \
datamodule.nt_train=50 \
datamodule.res_train=64 \
datamodule.nt_val=50 \
datamodule.res_val=64 \
datamodule.nt_test=50 \
datamodule.res_test=64 \
datamodule.batch_size=8 \
datamodule.samples=256 \
datamodule.train_regular=True \
model.params.time_slice=10 \
trainer.max_epochs=250 \
trainer.gpus=2 \
trainer.strategy='ddp'