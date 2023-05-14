import os
import os.path as osp

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# import internal libs
from .plot import plot_curves
from .utils import AverageMeter, update_lr, LambdaLayer
from .adv_attack import pgd_attack


def train_regression_model_adv(
        args, model, train_loader, test_loader,
        adv_step_size, adv_epsilon, adv_n_step,
        adv_bound_max=None, adv_bound_min=None,
        normalize_fn=None, denormalize_fn=None,
):

    print(model)
    model = model.to(args.gpu_id)

    # define loss function
    criterion = nn.MSELoss()

    if "model.pt" in os.listdir(args.save_root):
        print("The model has existed in model path '{}'. Load pretrained model.".format(args.save_root))
        model.load_state_dict(torch.load(os.path.join(args.save_root, "model.pt")))
        # evaluate the performance of the model
        eval_dict = eval_regression_model(model, test_loader, criterion)
        return eval_dict

    print("The model doen't exist in model path '{}'. Train a model with new settings.".format(args.save_root))

    # define the optimizer
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.train_lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # set the decay of learning rate
    lr_list = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.n_epoch)

    # define the train_csv
    learning_csv = os.path.join(args.save_root, "learning.csv")
    # define the res dict
    res_dict = {
        'train-loss-clean': [], 'train-loss-adv': [], 'test-loss': []
    }

    # starts training
    for epoch in range(args.n_epoch):
        # set the lr
        update_lr(optimizer, lr_list[epoch])

        # train the model
        model.train()
        train_dict = train_regression_model_adv_epoch(
            model, train_loader, optimizer, criterion,
            adv_step_size, adv_epsilon, adv_n_step,
            adv_bound_max, adv_bound_min,
            normalize_fn, denormalize_fn
        )
        train_loss_adv = train_dict["adv_loss"]
        train_loss_clean = train_dict["clean_loss"]

        # eval on the test set
        model.eval()
        with torch.no_grad():
            eval_dict = eval_regression_model(model, test_loader, criterion)
        test_loss = eval_dict["loss"]

        # save the res in dict
        res_dict['train-loss-clean'].append(train_loss_clean)
        res_dict['train-loss-adv'].append(train_loss_adv)
        res_dict['test-loss'].append(test_loss)
        # store the res in csv
        pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index=False)

        # show loss
        print('On train set - Epoch: {} \t Loss (clean): {:.6f} \t Loss (adv): {:.6f}'
              .format(epoch, train_loss_clean, train_loss_adv))
        print('On test set - Epoch: {} \t Loss: {:.6f}'
              .format(epoch, test_loss))

        if epoch % 10 == 0 or epoch == args.n_epoch - 1:
            # draw the curves
            plot_curves(args.save_root, res_dict)

    # save the model
    torch.save(model.cpu().state_dict(), os.path.join(args.save_root, "model.pt"))
    print("The model has been trained and saved in model path '{}'.".format(args.save_root))

    return model


def train_regression_model_adv_epoch(
        model, train_loader, optimizer, criterion,
        adv_step_size, adv_epsilon, adv_n_step,
        adv_bound_max=None, adv_bound_min=None,
        normalize_fn=None, denormalize_fn=None,
):
    model.train()
    device = next(model.parameters()).device

    if normalize_fn is not None:
        model_adv = nn.Sequential(
            LambdaLayer(normalize_fn),
            model,
        )
    else:
        model_adv = model

    if denormalize_fn is None:
        denormalize_fn = lambda x: x

    clean_loss_avg, adv_loss_avg = AverageMeter(), AverageMeter()

    for i, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        bs = batch_X.shape[0]

        # generate adversarial samples
        batch_X_adv = pgd_attack(
            model=model_adv, x=denormalize_fn(batch_X), y=batch_y,
            loss_func=lambda out, gt: criterion(out.squeeze(1), gt),
            step_size=adv_step_size, epsilon=adv_epsilon, n_step=adv_n_step,
            bound_max=adv_bound_max, bound_min=adv_bound_min
        )
        if normalize_fn is not None:
            batch_X_adv = normalize_fn(batch_X_adv)

        # train the model based on adversarial samples
        model.train()
        optimizer.zero_grad()
        outputs_adv = model(batch_X_adv).squeeze()
        loss = criterion(outputs_adv, batch_y)
        loss.backward()
        optimizer.step()

        # evaluate model performance on clean samples
        with torch.no_grad():
            model.eval()
            outputs_clean = model(batch_X).squeeze()
            loss_clean = criterion(outputs_clean, batch_y)

        # store stats
        adv_loss_avg.update(loss.item(), bs)
        clean_loss_avg.update(loss_clean.item(), bs)

    result_dict = {
        "clean_loss": clean_loss_avg.avg,
        "adv_loss": adv_loss_avg.avg
    }

    return result_dict


def eval_regression_model(model, test_loader, criterion):
    device = next(model.parameters()).device
    # evaluate the performance of the model
    model.eval()
    loss_avg = AverageMeter()
    with torch.no_grad():
        for step, (batch_X, batch_y) in enumerate(test_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            bs = batch_X.shape[0]

            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss_avg.update(loss.item(), bs)

    result_dict = {
        "loss": loss_avg.avg
    }

    return result_dict


def eval_regression_model_adv(
        model, test_loader, criterion,
        adv_step_size, adv_epsilon, adv_n_step,
        adv_bound_max=None, adv_bound_min=None,
        normalize_fn=None, denormalize_fn=None,
):
    """
        Evaluate the model on adversarial samples
    """
    device = next(model.parameters()).device
    # evaluate the performance of the model
    model.eval()

    if normalize_fn is not None:
        model_adv = nn.Sequential(
            LambdaLayer(normalize_fn),
            model,
        )
    else:
        model_adv = model

    if denormalize_fn is None:
        denormalize_fn = lambda x: x

    clean_loss_avg, adv_loss_avg = AverageMeter(), AverageMeter()
    for step, (batch_X, batch_y) in enumerate(test_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        bs = batch_X.shape[0]

        # generate adversarial samples
        batch_X_adv = pgd_attack(
            model=model_adv, x=denormalize_fn(batch_X), y=batch_y,
            loss_func=lambda out, gt: criterion(out.squeeze(1), gt),
            step_size=adv_step_size, epsilon=adv_epsilon, n_step=adv_n_step,
            bound_max=adv_bound_max, bound_min=adv_bound_min
        )
        if normalize_fn is not None:
            batch_X_adv = normalize_fn(batch_X_adv)

        with torch.no_grad():
            outputs_clean = model(batch_X).squeeze(1)
            outputs_adv = model(batch_X_adv).squeeze(1)
            loss_clean = criterion(outputs_clean, batch_y)
            loss_adv = criterion(outputs_adv, batch_y)

        clean_loss_avg.update(loss_clean.item(), bs)
        adv_loss_avg.update(loss_adv.item(), bs)

    result_dict = {
        "clean_loss": clean_loss_avg.avg,
        "adv_loss": adv_loss_avg.avg
    }

    return result_dict