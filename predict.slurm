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
pred_data_file="test.parquet"
pred_path="result.txt"

feat="SIKU-BERT"
method="blstm.crf"
batchsize=512

# predict on Test set
python -u run.py predict \
    -p \
    --streaming \
    --feat=$feat \
    --data=$data \
    --pred_data=$data/$pred_data_file \
    --pred_path=TestPredict/$pred_path \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.gram.nopunc.clr.llm \
    > log/pred/$feat.$method.gram.llm.nopunc.pred.clr.log 2>&1
