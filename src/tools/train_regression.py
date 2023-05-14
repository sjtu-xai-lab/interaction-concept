import os

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# import internal libs
from .plot import plot_curves
from .utils import AverageMeter, update_lr


def train_regression_model(args, model, train_loader, test_loader):

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
        'train-loss': [], 'test-loss': []
    }

    # starts training
    for epoch in range(args.n_epoch):
        # set the lr
        update_lr(optimizer, lr_list[epoch])

        # train the model
        model.train()
        train_dict = train_regression_model_epoch(
            model, train_loader, optimizer, criterion
        )
        train_loss = train_dict["loss"]

        # eval on the test set
        model.eval()
        with torch.no_grad():
            eval_dict = eval_regression_model(model, test_loader, criterion)
        test_loss = eval_dict["loss"]

        # save the res in dict
        res_dict['train-loss'].append(train_loss)
        res_dict['test-loss'].append(test_loss)
        # store the res in csv
        pd.DataFrame.from_dict(res_dict).to_csv(learning_csv, index=False)

        # show loss
        if epoch % 10 == 0 or epoch == args.n_epoch - 1:
            print('On train set - Epoch: {} \t Loss: {:.6f}'
                  .format(epoch, train_loss))
            print('On test set - Epoch: {} \t Loss: {:.6f}'
                  .format(epoch, test_loss))

        if epoch % 100 == 0 or epoch == args.n_epoch - 1:
            # draw the curves
            plot_curves(args.save_root, res_dict)

    # save the model
    torch.save(model.cpu().state_dict(), os.path.join(args.save_root, "model.pt"))
    print("The model has been trained and saved in model path '{}'.".format(args.save_root))

    return model


def train_regression_model_epoch(model, train_loader, optimizer, criterion):
    device = next(model.parameters()).device
    train_loss_avg = AverageMeter()
    model.train()
    for step, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        bs = batch_X.shape[0]

        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss_avg.update(loss.item(), bs)

    result_dict = {
        "loss": train_loss_avg.avg,
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

