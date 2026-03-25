data="training_data.txt"

feat="SIKU-BERT"
method="blstm.crf"

batchsize=50

# 2024.3.30

CUDA_VISIBLE_DEVICES=6 nohup python -u run.py train_single \
    -p \
    --base_model=../sinonom-ss/pretrained/sikubert \
    --feat=$feat \
    --data=dataset/$data \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.crf_last.nopunc.clr.llm \
    > log/train/$feat.$method.crf_last.llm.nopunc.clr.log 2>&1 &
