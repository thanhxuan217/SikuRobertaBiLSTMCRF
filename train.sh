data="training_data.txt"

feat="SIKU-BERT"
method="blstm.crf"

batchsize=50

# 训练gram
CUDA_VISIBLE_DEVICES=7 nohup python -u run.py train \
    -p \
    --base_model=../sinonom-ss/pretrained/sikubert \
    --feat=$feat \
    --data=dataset/$data \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.gram.nopunc.clr.llm \
    > log/train/$feat.$method.gram.llm.nopunc.clr.log 2>&1 &
