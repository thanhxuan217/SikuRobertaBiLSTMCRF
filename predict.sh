data="training_data.txt"
pred_data_file="EvaHan2024_testset.txt"
pred_data_file_zz="EvaHan2024_testsetzz.txt"
pred_path="result.a.txt"
pred_path_zz="result.zz.a.txt"

feat="SIKU-BERT"
method="blstm.crf"


batchsize=512
# 2024.3.30
# One crf

CUDA_VISIBLE_DEVICES=4 nohup python -u run.py predict \
    -p \
    --feat=$feat \
    --data=dataset/$data \
    --pred_data=dataset/$pred_data_file \
    --pred_path=TestPredict/$pred_path \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.crf_last.nopunc.clr.llm \
    > log/pred/$feat.$method.crf_last.llm.nopunc.pred.clr.a.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python -u run.py predict \
    -p \
    --feat=$feat \
    --data=dataset/$data \
    --pred_data=dataset/$pred_data_file_zz \
    --pred_path=TestPredict/$pred_path_zz \
    --batch_size=$batchsize \
    -f=exp/$feat.$method.crf_last.nopunc.clr.llm \
    > log/pred/$feat.$method.crf_last.llm.nopunc.pred.clr.zz.a.log 2>&1 &

