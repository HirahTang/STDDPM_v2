#!/bin/bash
#SBATCH --job-name=EMD
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=hendrixgpu06fl
#SBATCH --output=EMD_nonequi_10_0716.out
nvidia-smi
python main_qm9.py --n_epochs 3000 --exp_name STDDPM_nonequi_indexed_1_step_focus_on_smaller_steps --model gnn_dynamics --data_augmentation 1 --start_epoch 267  \
    --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 \
    --batch_size 64 --nf 128 --n_layers 6 --lr 1e-4 --normalize_factors [1,4,10] \
    --resume /home/qcx679/hantang/STDDPM_v2/outputs/STDDPM_nonequi_indexed_1_step_focus_on_smaller_steps \
    --test_epochs 20 --ema_decay 0.9999 --dataset dynamic --num_workers 8 --no_wandb