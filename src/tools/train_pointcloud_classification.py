import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from .plot import plot_curves
from .utils import AverageMeter, update_lr


def train_pointcloud_classification_model(args, model, train_loader, test_loader):

    print(model)
    model_type = model.__class__.__name__
    print(f"Model Type: {model_type}")
    model = model.to(args.gpu_id)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    if "model.pt" in os.listdir(args.save_root):
        print("The model has existed in model path '{}'. Load pretrained model.".format(args.save_root))
        model.load_state_dict(torch.load(os.path.join(args.save_root, "model.pt")))

        # evaluate the performance of the model
        eval_dict = eval_pointcloud_classification_model(model, test_loader, criterion)

        return eval_dict

    print("The model doen't exist in model path '{}'. Train a model with new settings.".format(args.save_root))

    # # define the optimizer and the scheduler
    # if model_type == "PointNet":
    #     optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # else:
    #     raise NotImplementedError

    # define the optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # set the decay of learning rate
    lr_list = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.n_epoch)

    # define the train_csv
    learning_csv = os.path.join(args.save_root, "learning.csv")
    # define the res dict
    res_dict = {
        'train-loss': [], 'train-cls-loss': [], 'train-reg-loss': [], 'train-acc': [],
        'test-loss': [], "test-acc": []
    }

    t_log = time.time()
    t_plot = time.time()
    best_test_acc = 0.
    # starts training
    for epoch in range(args.n_epoch):

        # set the lr
        update_lr(optimizer, lr_list[epoch])

        # train the model
        model.train()
        train_dict = train_pointcloud_classification_model_epoch(
            model, train_loader, optimizer, criterion, args.gpu_id
        )
        train_loss = train_dict["loss"]
        train_acc = train_dict["acc"]
        train_reg_loss = train_dict["reg_loss"]
        train_cls_loss = train_dict["cls_loss"]

        # eval on test set
        model.eval()
        with torch.no_grad():
            eval_dict = eval_pointcloud_classification_model(
                model, test_loader, criterion
            )
            test_loss = eval_dict["loss"]
            test_acc = eval_dict["acc"]

        # scheduler.step()

        # save the res in dict
        res_dict['train-loss'].append(train_loss)
        res_dict["train-acc"].append(train_acc)
        res_dict['test-loss'].append(test_loss)
        res_dict["test-acc"].append(test_acc)
        res_dict['train-reg-loss'].append(train_reg_loss)
        res_dict['train-cls-loss'].append(train_cls_loss)
        # store the res in csv
        pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index=False)

        # show loss
        if epoch % 10 == 0 or epoch == args.n_epoch or time.time() - t_log > 30:
            print('On train set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                  .format(epoch, train_loss, train_acc))
            print('On test set - Epoch: {} \t Loss: {:.6f} \t Acc: {:.9f}'
                  .format(epoch, test_loss, test_acc))
            t_log = time.time()

        if epoch % 100 == 0 or epoch == args.n_epoch - 1 or time.time() - t_plot > 60:
            # draw the curves
            plot_curves(args.save_root, res_dict)
            t_plot = time.time()

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), osp.join(args.save_root, "model_best.pt"))
            print("Save best at '{}'.".format(args.save_root))

    plot_curves(args.save_root, res_dict)
    # save the model
    torch.save(model.cpu().state_dict(), osp.join(args.save_root, "model.pt"))
    print("The model has been trained and saved in model path '{}'.".format(args.save_root))

    return model


def train_pointcloud_classification_model_epoch(model, train_loader, optimizer, criterion, device):
    model_type = model.__class__.__name__
    train_loss_avg, train_acc_avg = AverageMeter(), AverageMeter()
    train_cls_loss_avg, train_reg_loss_avg = AverageMeter(), AverageMeter()
    model.train()
    for step, (batch_X, batch_y) in enumerate(tqdm(train_loader, desc="training", ncols=100)):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        bs = batch_X.shape[0]

        optimizer.zero_grad()
        if model_type == "PointNet":
            outputs, trans_feat = model(batch_X)
            loss_reg = model.get_reg_loss(trans_feat)
        else:
            outputs = model(batch_X)
            loss_reg = torch.tensor(0)
        loss_cls = criterion(outputs, batch_y)
        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=-1)
        acc = torch.sum(preds == batch_y) / bs

        train_loss_avg.update(loss.item(), bs)
        train_cls_loss_avg.update(loss_cls.item(), bs)
        train_reg_loss_avg.update(loss_reg.item(), bs)
        train_acc_avg.update(acc.item(), bs)

    result_dict = {
        "loss": train_loss_avg.avg,
        "reg_loss": train_reg_loss_avg.avg,
        "cls_loss": train_cls_loss_avg.avg,
        "acc": train_acc_avg.avg
    }

    return result_dict


def eval_pointcloud_classification_model(model, test_loader, criterion):
    device = next(model.parameters()).device
    # evaluate the performance of the model
    model.eval()
    loss_avg, acc_avg = AverageMeter(), AverageMeter()
    model_type = model.__class__.__name__
    with torch.no_grad():
        for step, (batch_X, batch_y) in enumerate(tqdm(test_loader, desc="testing", ncols=100)):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            bs = batch_X.shape[0]
            if model_type == "PointNet":
                outputs, trans_feat = model(batch_X)
            else:
                outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=-1)

            loss = criterion(outputs, batch_y)
            acc = torch.sum(preds == batch_y) / bs

            loss_avg.update(loss.item(), bs)
            acc_avg.update(acc.item(), bs)

    # print('On test set - \t Loss: {:.6f} \t Acc: {:.9f}'.format(loss_avg.avg, acc_avg.avg))
    result_dict = {
        "loss": loss_avg.avg,
        "acc": acc_avg.avg
    }

    return result_dict
