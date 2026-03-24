# -*- coding: utf-8 -*-
# @Time    : 2024/1/22 19:14

import argparse
import os
import torch
from datetime import datetime
import numpy as np
import random
from parsering.config import Config

from parsering.cmd.train_gram import Train
from parsering.cmd.train_single import Train_single
from parsering.cmd.predict_gram import Predict
from parsering.cmd.predict_single import Predict_single


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Gpt2 Incremental model.'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'train': Train(),
        'train_single': Train_single(),
        'predict': Predict(),
        'predict_single': Predict_single()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--conf', '-c', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini'),
                               help='path to config file')
        subparser.add_argument('--file', '-f', default='exp/evahan',
                               help='path to saved files')
        subparser.add_argument('--preprocess', '-p', action='store_true',
                               help='whether to preprocess the data first')
        subparser.add_argument('--device', '-d', default='1',
                               help='ID of GPU to use')
        subparser.add_argument('--seed', '-s', default=1, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=8, type=int,
                               help='max num of threads')
        subparser.add_argument('--feat', default=None,
                               choices=['SIKU-BERT'],
                               help='choices of embedding model')
        subparser.add_argument('--joint', action='store_true',
                               help='whether to train jointly')
        subparser.add_argument('--batch_size', default=32, type=int,
                               help='batch size')
        subparser.add_argument('--task', default='segmentation',
                               choices=['segmentation', 'punctuation'],
                               help='the task type to run')

    args = parser.parse_args()
    print(f'NOTE: {args.feat} 模型')
    print(f'{datetime.now()}')
    print(f'The process id is {os.getpid()}')
    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    print(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    seed_torch(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.save_model = os.path.join(args.file, 'model.pth')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = Config(args.conf).update(vars(args))
    
    # Resolve base_model path relative to the script directory if it's a relative path
    if hasattr(args, 'base_model') and args.base_model and not os.path.isabs(args.base_model):
        args.base_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.base_model)
        
    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(args)
