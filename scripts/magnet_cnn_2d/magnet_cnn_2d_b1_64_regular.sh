#!/bin/bash

#SBATCH --job-name=magnet_cnn_2d_b1_64_regular
#SBATCH --partition=long                        
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:2      
#SBATCH --mem=60G                                     
#SBATCH --time=3:30:00
#SBATCH --array=1-5  
#SBATCH -o /network/scratch/o/oussama.boussif/slurms/magnet_cnn_2d_b1_64_regular-slurm-%A_%a.out  

param_store=scripts/seeds.txt
seed=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')
# 1. Load the required modules
module --quiet load anaconda/3
conda activate dedalus

python run.py \
seed=$seed \
model=magnet_cnn_2d \
name=magnet_cnn_2d \
datamodule=h5_datamodule_implicit_2d \
datamodule.train_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B1/burgers_train_B1_64.h5' \
datamodule.val_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B1/burgers_test_B1_64.h5' \
datamodule.test_path='/home/mila/o/oussama.boussif/scratch/pdeone/data/B1/burgers_test_B1_64.h5' \
datamodule.nt_train=50 \
datamodule.res_train=64 \
datamodule.nt_val=50 \
datamodule.res_val=64 \
datamodule.nt_test=50 \
datamodule.res_test=64 \
datamodule.samples=256 \
model.params.time_slice=10 \
model.params.teacher_forcing=True \
trainer.gpus=2 \
trainer.strategy='ddp' \
trainer.max_epochs=250