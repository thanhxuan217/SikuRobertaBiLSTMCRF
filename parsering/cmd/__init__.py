# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 20:29
# @Author  : wxb
# @File    : __init__.py
"""
File khởi tạo module `cmd`. 
Đóng gói và export các module đại diện cho luồng Training/Prediction 
(dành cho cả hai-CRF và single-CRF) 
để các file khởi chạy main có thể import dễ dàng.
"""


from .train_gram import Train
from .train_single import Train_single

from .predict_gram import Predict
from .predict_single import Predict_single

__all__ = ['Train', 'Train_single',
           'Predict', 'Predict_single']
