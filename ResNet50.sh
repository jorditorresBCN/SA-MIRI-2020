#!/bin/bash
#SBATCH --job-name="ResNet50"
#SBATCH -D .
#SBATCH --output=ResNet50_%j.output
#SBATCH --error=ResNet50_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00

module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML

python ResNet50.py --epochs 5 --batch_size 256 --n_gpus 1
python ResNet50.py --epochs 5 --batch_size 512 --n_gpus 2
python ResNet50.py --epochs 5 --batch_size 1024 --n_gpus 4

