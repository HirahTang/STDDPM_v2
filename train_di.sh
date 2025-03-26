#!/bin/bash
#SBATCH --job-name=EMD
#SBATCH --ntasks=1 --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=EMD_2.out
nvidia-smi
python main_qm9.py --n_epochs 3000 --exp_name STDDPM_equi --start_epoch 0 \
    --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 \
    --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] \
    --test_epochs 20 --ema_decay 0.9999 --dataset dynamic
