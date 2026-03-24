#!/bin/bash
#SBATCH --job-name=sikubert_cnn
#SBATCH --output=softmax_logs/slurm_%j.out
#SBATCH --error=softmax_logs/slurm_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=24:00:00

source /media02/ddien02/datnd/miniconda3/etc/profile.d/conda.sh
conda activate /media02/ddien02/thanhxuan217/envs/sikubert

data="/data"
feat="SIKU-BERT"
method="blstm.crf"
batchsize=50
mkdir -p log/train exp

python -u run.py train_single \
    -p \
    --feat=$feat \
    --data=$data \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.crf_last.nopunc.clr.llm \
    > log/train/$feat.$method.crf_last.llm.nopunc.clr.log 2>&1
