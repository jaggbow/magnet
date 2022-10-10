#!/bin/bash

#SBATCH --job-name=mpnn_2d_b1_64_irregular_concentrated
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1      
#SBATCH --mem=60G                                     
#SBATCH --time=1:30:00
#SBATCH --array=1-5  
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/mpnn_2d_b1_64_irregular_concentrated-slurm-%A_%a.out  

param_store=scripts/new_seeds.txt
seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
# 1. Load the required modules
module --quiet load anaconda/3
conda activate dedalus

python run.py \
seed=$seed \
model=mpnn_2d \
name=mpnn_2d \
datamodule=h5_datamodule_graph_2d \
datamodule.train_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B1/concentrated/burgers_train_irregular_B1_64.h5' \
datamodule.val_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B1/burgers_test_B1_32.h5' \
datamodule.test_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B1/burgers_test_B1_32.h5' \
datamodule.nt_train=50 \
datamodule.res_train=64 \
datamodule.nt_val=50 \
datamodule.res_val=32 \
datamodule.nt_test=50 \
datamodule.res_test=32 \
datamodule.batch_size=32 \
model.params.time_window=10 \
model.params.neighbors=4 \
model.params.teacher_forcing=False \
trainer.max_epochs=250 \
datamodule.train_regular=False