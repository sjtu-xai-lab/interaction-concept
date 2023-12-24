import os
import os.path as osp
import re
import argparse
import torch
import torch.nn as nn

from datasets.tabular import TabularDataset, TabularDatasetCorrupted
import models.tabular as models
from tools.train import train_model
from setup_exp import setup_train_model_tabular


print("-------Parsing program arguments--------")
parser = argparse.ArgumentParser(description="train model code")
parser.add_argument('--data-root', default='/data2/lmj/data/tabular', type=str,
                    help="root folder for dataset.")
parser.add_argument('--gpu-id', default=0, type=int, help="set the device.")
parser.add_argument("--dataset", default="census", type=str,
                    help="set the dataset used: commercial, census, bike")
parser.add_argument("--arch", default="resmlp5", type=str,
                    help="the network architecture: mlp5, resmlp5")
parser.add_argument("--save-root", default="../saved-models", type=str,
                    help='the path of pretrained model.')

# set the batch size for training
parser.add_argument('--batch-size', default=512, type=int, help="set the batch size for training.")
# set the learning rate for training
parser.add_argument('--lr', default=0.01, type=float, help="set the learning rate for training.")
# set the decay of learning rate
parser.add_argument("--logspace", default=1, type=int,
                    help='the decay of learning rate. if set as 1, then lr will decay exponentially '
                         'for 10x over the training process.')
# set the number of epochs for training model.
parser.add_argument("--n-epoch", default=500, type=int, help='the number of epochs for training model.')
# set the model seed
parser.add_argument("--seed", default=0, type=int, help="set the seed used for training model.")
# optional: whether to rebalance the data
parser.add_argument("--balance", action="store_true", help="Set this flag if you need data re-sampling for "
                                                           "class-imbalanced datasets (e.g. census).")
args = parser.parse_args()
setup_train_model_tabular(args)

# ===============================================
#   prepare the dataset (for train & eval)
# ===============================================
print("-----------preparing dataset-----------")
print("dataset - {}".format(args.dataset))
dataset = TabularDataset(args.data_root, args.dataset)
train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size, balance=args.balance)

model = models.__dict__[args.arch](**args.model_kwargs)
model = model.to(args.gpu_id)

# ===============================================
#   train the model
# ===============================================
print("------------preparing model------------")
train_model(args, model, train_loader, test_loader, task=args.task)


