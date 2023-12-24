import os
import os.path as osp
import argparse
import torch
import torch.nn as nn

from datasets.get_dataset import get_dataset
import models.image_tiny as models
from tools.train import train_model
from setup_exp import setup_train_model_image_tiny


parser = argparse.ArgumentParser(description="train model code")
parser.add_argument('--data-root', default='/data2/lmj/data', type=str,
                    help="root folder for dataset.")
parser.add_argument('--gpu-id', default=0, type=int, help="set the device.")
parser.add_argument("--dataset", default="mnist", type=str,
                    help="set the dataset used: mnist")
parser.add_argument("--arch", default="lenet", type=str,
                    help="the network architecture: lenet")
parser.add_argument("--save-root", default="../saved-models", type=str,
                    help='the path of pretrained model.')

parser.add_argument('--batch-size', default=128, type=int, help="set the batch size for training.")
parser.add_argument('--lr', default=0.01, type=float, help="set the learning rate for training.")
parser.add_argument("--logspace", default=1, type=int,
                    help='the decay of learning rate. if set as 1, then lr will decay exponentially '
                         'for 10x over the training process.')
parser.add_argument("--n-epoch", default=50, type=int, help='the number of epochs for training model.')
parser.add_argument("--seed", default=0, type=int, help="set the seed used for training model.")
args = parser.parse_args()
setup_train_model_image_tiny(args)

# ===============================================
#   prepare the dataset (for train & eval)
# ===============================================
print("dataset - {}".format(args.dataset))
dataset = get_dataset(args.data_root, args.dataset)
train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size)

model = models.__dict__[args.arch](**args.model_kwargs)
model = model.to(args.gpu_id)

# ===============================================
#   train the model
# ===============================================
train_model(args, model, train_loader, test_loader, task=args.task)


