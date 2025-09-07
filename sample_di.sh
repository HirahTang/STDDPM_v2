#!/bin/bash
#SBATCH --job-name=EMD
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=EMD_dynamic_100.out
nvidia-smi
python eval_sample_dynamic.py --model_path /home/qcx679/hantang/STDDPM_v2/outputs/STDDPM_nonequi_indexed_1_step_focus_on_smaller_steps \
    --probabilistic_model dynamic --dynamic_t 400 --markovian_sampling
