import json
import os
import os.path as osp
import re
from pprint import pprint
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Union, Tuple, Dict
import models.tabular as models
from datasets.tabular import TabularDataset
from harsanyi import AndOrHarsanyi, AndOrHarsanyiSparsifier
from harsanyi.aog_inference import reorganize_and_or_harsanyi
from baseline_values import get_baseline_value
from tools.utils import set_seed
from tools.train import eval_model
from setup_exp import setup_eval_interaction_tabular


def parse_args():
    parser = argparse.ArgumentParser(description="evaluate the iou (on image datasets)")
    parser.add_argument('--data-root', default='/data2/lmj/data/tabular', type=str,
                        help="root folder for dataset.")
    parser.add_argument("--model-args", type=str, default=None,
                        help="hyper-parameters for the pre-trained model")
    parser.add_argument('--gpu-id', default=0, type=int, help="set the device.")
    parser.add_argument("--model-root", default="../saved-models", type=str,
                        help='the root folder that stores the pre-trained model')
    parser.add_argument("--save-root", default="../saved-results", type=str,
                        help='the root folder to save results')

    parser.add_argument("--baseline", type=str, default="zero",
                        help="configuration of the baseline value")
    parser.add_argument("--selected-dim", type=str, default="gt-log-odds",
                        help="use which dimension to compute interactions")

    parser.add_argument("--selected-classes", type=str, default=None,
                        help="choose only samples from specific classes for analysis")
    parser.add_argument("--n-sample-each-class", type=int, default=None,
                        help="the number of samples from each class")

    # please specify the following parameters if you use and-or interaction
    parser.add_argument("--sparsify-loss", default=None, type=str,
                        help="use which type of loss to sparsify and or interactions: l1 | l1_on_xxx "
                             "Commonly used: l1")
    parser.add_argument("--sparsify-qthres", default=None, type=float,
                        help="the threshold to bound the magnitude of q: q in [-thres*std, thres*std]. "
                             "This should be a float number, commly used: 0.02")
    parser.add_argument("--sparsify-qstd", default=None, type=str,
                        help="the standard to bound the magnitude of q: q in [-thres*std, thres*std]. "
                             "Choose from: vS, vS-v0, vN-v0, none. Commonly used: vN-v0")
    parser.add_argument("--sparsify-lr", default=None, type=float,
                        help="the learning rate to learn (p and q). Commonly used: depends.")
    parser.add_argument("--sparsify-niter", default=None, type=int,
                        help="how many iteractions to optimize (p and q). Commonly used: 20000, 50000")

    args = parser.parse_args()
    setup_eval_interaction_tabular(args)
    return args


def get_model(args):
    model = models.__dict__[args.arch](**args.model_kwargs)
    model = model.to(args.gpu_id)
    model.load_state_dict(torch.load(osp.join(args.model_root, args.model_args, "model.pt"),
                                     map_location=torch.device(f"cuda:{args.gpu_id}")))
    model.eval()
    return model


def get_correct_sample_indices(net: nn.Module, samples: torch.Tensor, labels: torch.Tensor, task: str, bs: int = 1):
    device = next(net.parameters()).device
    n_sample = samples.shape[0]
    n_batch = int(np.ceil(n_sample / bs))
    correct_indices = []
    if task == "classification":
        for i in range(n_batch):
            X_batch = samples[i*bs:(i+1)*bs].clone()
            y_batch = labels[i*bs:(i+1)*bs].clone()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output_batch = net(X_batch)
            pred_batch = torch.argmax(output_batch, dim=-1)
            correct_indices.append(torch.arange(i*bs, i*bs+X_batch.shape[0])[pred_batch == y_batch])
    else:
        raise NotImplementedError

    return torch.cat(correct_indices).tolist()


def get_sample_indices_each_class(sample_indices: List, labels: torch.Tensor, n_sample_each_class: int = 200, selected_classes=None):
    if n_sample_each_class is None:
        n_sample_each_class = 200
    if selected_classes is None:
        selected_classes = sorted(torch.unique(labels).tolist())

    slots = {class_id: n_sample_each_class for class_id in torch.unique(labels).tolist()}
    for class_id in slots.keys():
        if class_id not in selected_classes:
            slots[class_id] = 0

    selected = {class_id: [] for class_id in selected_classes}
    print(slots)
    for i in sample_indices:
        label = labels[i].item()
        if slots[label] > 0:
            slots[label] -= 1
            selected[label].append(i)
        if sum(list(slots.values())) == 0:
            break
    return selected


def evaluate_single(
        forward_func, selected_dim,
        sample, baseline, label,
        sparsify_kwargs, save_folder
):
    device = sample.device
    if osp.exists(osp.join(save_folder, "I_and.pth")) and osp.exists(osp.join(save_folder, "I_or.pth")):
        print("load previous")
        I_and = torch.load(osp.join(save_folder, "I_and.pth"), map_location=device)
        I_or = torch.load(osp.join(save_folder, "I_or.pth"), map_location=device)
        return torch.cat([I_and, I_or])

    assert sample.shape[0] == 1
    _, d = sample.shape
    attributes = [r"$x_{" + str(i) + r"}$" for i in range(d)]

    # 1. calculate interaction
    calculator = AndOrHarsanyi(
        model=forward_func, selected_dim=selected_dim,
        x=sample, baseline=baseline, y=label, calc_bs=None, verbose=0
    )
    with torch.no_grad():
        calculator.attribute()
        masks = calculator.get_masks()
        I_and_, I_or_ = calculator.get_interaction()
        calculator.save(save_folder=osp.join(save_folder, "before_sparsify"))

    with open(osp.join(save_folder, "log.txt"), 'w') as f:  # 检查 efficiency 性质
        with torch.no_grad(): f.write(f"output: {forward_func(sample)}\n")
        f.write("\n[Before Sparsifying]\n")
        f.write("sum of I^and:\n")
        f.write(f"\t{I_and_.sum()}\n")
        f.write("sum of I^or:\n")
        f.write(f"\t{I_or_.sum()}\n")
        f.write(f"\tSum of I^and and I^or:\n"
                f"\t{torch.sum(I_and_) + torch.sum(I_or_)}\n")
        f.write("\n")

    sparsifier = AndOrHarsanyiSparsifier(calculator=calculator, **sparsify_kwargs)
    sparsifier.sparsify(verbose_folder=osp.join(save_folder, "sparsify_verbose"))
    with torch.no_grad():
        I_and, I_or = sparsifier.get_interaction()
        I_and, I_or = reorganize_and_or_harsanyi(masks, I_and, I_or)
        sparsifier.save(save_folder=osp.join(save_folder, "after_sparsify"))
    torch.save(I_and, osp.join(save_folder, "I_and.pth"))
    torch.save(I_or, osp.join(save_folder, "I_or.pth"))
    with open(osp.join(save_folder, "log.txt"), 'a') as f:  # 检查 efficiency 性质
        f.write("\n[After Sparsifying]\n")
        f.write(f"\tSum of I^and and I^or: {torch.sum(I_and) + torch.sum(I_or)}\n")

    return torch.cat([I_and, I_or])


if __name__ == '__main__':
    args = parse_args()
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_path = osp.join(args.save_root, f"log_{current_time}.txt")

    # =========================================
    #     initialize the model and dataset
    # =========================================
    model = get_model(args)
    dataset = TabularDataset(args.data_root, args.dataset)
    train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size, shuffle_train=False)
    X_train, y_train, X_test, y_test = dataset.get_data()
    X_all = torch.cat([X_train, X_test])
    y_all = torch.cat([y_train, y_test])

    # =========================================
    #    validate the pre-trained model
    # =========================================
    # train_eval_dict = eval_model(model, train_loader, task=args.task)
    test_eval_dict = eval_model(model, test_loader, task=args.task)
    with open(log_path, "w") as logfile:
        # print("train loss:", train_eval_dict, file=logfile)
        print("test loss:", test_eval_dict, file=logfile)

    # =========================================
    #     get correctly classified samples
    # =========================================
    with torch.no_grad():
        correct_indices = get_correct_sample_indices(model, X_all, y_all, task=args.task, bs=args.batch_size)
    selected_indices = get_sample_indices_each_class(correct_indices, y_all,
                                                     n_sample_each_class=args.n_sample_each_class,
                                                     selected_classes=args.selected_classes)
    with open(log_path, "a") as logfile:
        print("# of correct samples", len(correct_indices), file=logfile)
        print("# of selected samples each class",
              {i: len(selected) for i, selected in selected_indices.items()}, file=logfile)

    # =========================================
    #   evaluate interaction for each sample
    # =========================================
    for class_id in selected_indices.keys():

        for sample_id in selected_indices[class_id]:
            set_seed(args.seed)
            print(f"Class id: {class_id}, Sample id: {sample_id}")
            save_folder = osp.join(args.save_root, f"class_{class_id}", f"sample_{sample_id:>05d}")
            os.makedirs(save_folder, exist_ok=True)

            sample = X_all[sample_id].clone().unsqueeze(0).to(args.gpu_id)
            label = y_all[sample_id].clone().item()
            baseline = get_baseline_value(X_all, baseline_config=args.baseline).to(args.gpu_id)
            forward_func = model

            sparsify_kwargs = {
                "trick": "pq",
                "loss": args.sparsify_loss,
                "qthres": args.sparsify_qthres,
                "qstd": args.sparsify_qstd,
                "lr": args.sparsify_lr,
                "niter": args.sparsify_niter
            }
            I_and_or = evaluate_single(
                forward_func=forward_func, selected_dim=args.selected_dim,
                sample=sample, baseline=baseline, label=label,
                save_folder=save_folder, sparsify_kwargs=sparsify_kwargs,
            )


