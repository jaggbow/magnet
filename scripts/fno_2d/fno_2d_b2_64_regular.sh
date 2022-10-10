#!/bin/bash

#SBATCH --job-name=fno_2d_b2_64_regular
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1        
#SBATCH --mem=60G                                     
#SBATCH --time=2:30:00
#SBATCH --array=1-5  
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/fno_2d_b2_64_regular-slurm-%A_%a.out  

param_store=scripts/seeds.txt
seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
# 1. Load the required modules
module --quiet load anaconda/3
conda activate dedalus

python run.py \
seed=$seed \
model=fno_2d \
name=fno_2d \
datamodule=h5_datamodule_2d \
datamodule.train_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B2/burgers_train_B2_64.h5' \
datamodule.val_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B2/burgers_test_B2_64.h5' \
datamodule.test_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B2/burgers_test_B2_64.h5' \
datamodule.nt_train=50 \
datamodule.res_train=64 \
datamodule.nt_val=50 \
datamodule.res_val=64 \
datamodule.nt_test=50 \
datamodule.res_test=64 \
model.params.time_history=10 \
model.params.time_future=10 \
model.params.teacher_forcing=False \
model.params.modes_1=12 \
model.params.modes_2=12 \
trainer.max_epochs=250