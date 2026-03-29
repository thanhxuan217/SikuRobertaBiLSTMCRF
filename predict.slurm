#!/bin/bash
#SBATCH --job-name=pred_sikubert_blstm_crf
#SBATCH --output=logs/pred_%j.out
#SBATCH --error=logs/pred_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00

source /media02/ddien02/datnd/miniconda3/etc/profile.d/conda.sh
conda activate /media02/ddien02/thanhxuan217/envs/sikubert

data="/media02/ddien02/thanhxuan217/main_src/data_punc/test"
pred_data_file="punctuation_sliding_part_9.parquet"
pred_path="result.txt"

feat="SIKU-BERT"
method="blstm.crf"
batchsize=512

mkdir -p TestPredict

# predict on Test set (load LoRA-finetuned model)
python -u run.py predict \
    -p \
    --streaming \
    --feat=$feat \
    --data=$data \
    --pred_data=$data/$pred_data_file \
    --pred_path=TestPredict/$pred_path \
    --batch_size=$batchsize \
    --base_model="/media02/ddien02/thanhxuan217/main_src/pretrained/sikubert" \
    --use_qlora \
    -f=/media02/ddien02/thanhxuan217/main_src/exp/$feat.$method.use_qlora
