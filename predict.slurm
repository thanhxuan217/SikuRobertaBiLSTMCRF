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
pred_data_file="test"
pred_data_file_zz="test"
pred_path="result.a.txt"
pred_path_zz="result.zz.a.txt"

feat="SIKU-BERT"
method="blstm.crf"
batchsize=512

python -u run.py predict \
    -p \
    --feat=$feat \
    --data=$data \
    --pred_data=$data/$pred_data_file \
    --pred_path=TestPredict/$pred_path \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.crf_last.nopunc.clr.llm \
    > log/pred/$feat.$method.crf_last.llm.nopunc.pred.clr.a.log 2>&1

python -u run.py predict \
    -p \
    --feat=$feat \
    --data=$data \
    --pred_data=$data/$pred_data_file_zz \
    --pred_path=TestPredict/$pred_path_zz \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.crf_last.nopunc.clr.llm \
    > log/pred/$feat.$method.crf_last.llm.nopunc.pred.clr.zz.a.log 2>&1
