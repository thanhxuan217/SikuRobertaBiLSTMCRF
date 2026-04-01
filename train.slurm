#!/bin/bash
#SBATCH --job-name=sikubert_blstm_crf
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00

source /media02/ddien02/datnd/miniconda3/etc/profile.d/conda.sh
conda activate /media02/ddien02/thanhxuan217/envs/sikubert

data="/media02/ddien02/thanhxuan217/main_src/data_punc"
feat="SIKU-BERT"
method="blstm.crf"
batchsize=64
mkdir -p exp

python -u run.py train \
    -p \
    --max_steps="100000" \
    --eval_steps="5000" \
    --val_batches="500" \
    --save_steps="5000" \
    --device="0" \
    --feat=$feat \
    --base_model="/media02/ddien02/thanhxuan217/main_src/pretrained/sikubert" \
    --data=$data \
    --batch_size=$batchsize \
    --task="punctuation" \
    --use_qlora \
    -f=/media02/ddien02/thanhxuan217/main_src/exp/$feat.$method.use_qlora
