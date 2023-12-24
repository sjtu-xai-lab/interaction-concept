import argparse
import inspect
import os
import os.path as osp
from tools.utils import makedirs, set_seed
import json
import socket
import torch
import warnings
import re
from typing import List, Tuple, Dict


def _json_encoder_default(obj):
    if isinstance(obj, torch.Tensor):
        return str(obj)
    elif isinstance(obj, type(lambda x: x)):
        return inspect.getsource(obj).strip()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_args(args, save_path):
    with open(save_path, "w") as f:
        json.dump(vars(args), f, indent=4,
                  default=_json_encoder_default)


def generate_dataset_model_desc(args):
    return f"dataset={args.dataset}" \
           f"{'_balance' if 'balance' in vars(args) and args.balance else ''}" \
           f"_model={args.arch}" \
           f"_epoch={args.n_epoch}" \
           f"_bs={args.batch_size}" \
           f"_lr={args.lr}" \
           f"_logspace={args.logspace}" \
           f"_seed={args.seed}"


def generate_adv_train_desc(args):
    return f"step-size={args.adv_step_size}" \
           f"_epsilon={args.adv_epsilon}" \
           f"_n-step={args.adv_n_step}"


def parse_dataset_model_desc(dataset_model_desc: str) -> Dict:
    """
    Parse model args
    :param dataset_model_desc: the arg string
    :return: dict

    >>> parse_dataset_model_desc("dataset=census_balance_model=mlp5_epoch=5000_bs=512_lr=0.1_logspace=2_seed=0")
    {'dataset': 'census', 'arch': 'mlp5', 'balance': True, 'n_epoch': 5000, 'batch_size': 512, 'lr': 0.1, 'logspace': 2, 'seed': 0}

    >>> parse_dataset_model_desc("dataset=commercial_model=mlp5_epoch=5000_bs=512_lr=0.01_logspace=2_seed=0")
    {'dataset': 'commercial', 'arch': 'mlp5', 'balance': False, 'n_epoch': 5000, 'batch_size': 512, 'lr': 0.01, 'logspace': 2, 'seed': 0}

    >>> parse_dataset_model_desc("dataset=gaussian_rule_001_regression_10d_v1_model=mlp5_sigmoid_epoch=500_bs=512_lr=0.01_logspace=1_seed=0")
    {'dataset': 'gaussian_rule_001_regression_10d_v1', 'arch': 'mlp5_sigmoid', 'balance': False, 'n_epoch': 500, 'batch_size': 512, 'lr': 0.01, 'logspace': 1, 'seed': 0}

    >>> parse_dataset_model_desc("dataset=gaussian_rule_001_regression_10d_v1_model=mlp5_sigmoid_epoch=100_bs=512_lr=0.01_logspace=1_seed=0_step-size=0.01_epsilon=0.1_n-step=20")
    {'dataset': 'gaussian_rule_001_regression_10d_v1', 'arch': 'mlp5_sigmoid', 'balance': False, 'n_epoch': 100, 'batch_size': 512, 'lr': 0.01, 'logspace': 1, 'seed': 0, 'adv_step_size': 0.01, 'adv_epsilon': 0.1, 'adv_n_step': 20}
    """
    pattern = r"dataset=(.+)" \
              r"\_model=(.+)" \
              r"\_epoch=(.+)" \
              r"\_bs=(.+)" \
              r"\_lr=(.+)" \
              r"\_logspace=(.+)" \
              r"\_seed=(.+)" \
              r"\_step-size=(.+)" \
              r"\_epsilon=(.+)" \
              r"\_n-step=(.+)"
    match = re.match(pattern, dataset_model_desc)

    if match is not None:
        dataset, arch, n_epoch, batch_size, lr, logspace, seed, adv_step_size, adv_epsilon, adv_n_step = match.groups()
        balance = "_balance" in dataset
        if balance:
            dataset = "_".join(dataset.split("_")[:-1])
        n_epoch = int(n_epoch)
        batch_size = int(batch_size)
        lr = float(lr)
        logspace = int(logspace)
        seed = int(seed)
        adv_step_size = float(adv_step_size)
        adv_epsilon = float(adv_epsilon)
        adv_n_step = int(adv_n_step)

        return {
            "dataset": dataset,
            "arch": arch,
            "balance": balance,
            "n_epoch": n_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "logspace": logspace,
            "seed": seed,
            "adv_step_size": adv_step_size,
            "adv_epsilon": adv_epsilon,
            "adv_n_step": adv_n_step
        }
    else:
        pattern = r"dataset=(.+)" \
                  r"\_model=(.+)" \
                  r"\_epoch=(.+)" \
                  r"\_bs=(.+)" \
                  r"\_lr=(.+)" \
                  r"\_logspace=(.+)" \
                  r"\_seed=(.+)"
        match = re.match(pattern, dataset_model_desc)
        assert match is not None

        dataset, arch, n_epoch, batch_size, lr, logspace, seed = match.groups()
        balance = "_balance" in dataset
        if balance:
            dataset = "_".join(dataset.split("_")[:-1])
        n_epoch = int(n_epoch)
        batch_size = int(batch_size)
        lr = float(lr)
        logspace = int(logspace)
        seed = int(seed)  # TODO: add support for adversarial training

        return {
            "dataset": dataset,
            "arch": arch,
            "balance": balance,
            "n_epoch": n_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "logspace": logspace,
            "seed": seed,
        }


def generate_sparsified_and_or_desc(args) -> str:
    return f"loss={args.sparsify_loss}" \
           f"_qthres={args.sparsify_qthres}" \
           f"_qstd={args.sparsify_qstd}" \
           f"_lr={args.sparsify_lr}" \
           f"_niter={args.sparsify_niter}"


def _init_model_setting(args):
    if args.dataset in ["census"] and \
            args.arch in [*[f"mlp{i}" for i in range(2, 11)],
                          *[f"resmlp{i}" for i in range(2, 11)],]:
        args.in_dim = 12
        args.out_dim = 2
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif args.dataset in ["commercial"] and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 10
        args.out_dim = 2
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif args.dataset in ["yeast"] and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 8
        args.out_dim = 10
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif args.dataset in ["wine"] and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 11
        args.out_dim = 7
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif args.dataset in ["glass"] and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 9
        args.out_dim = 6
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif args.dataset in ["telescope"] and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 10
        args.out_dim = 2
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif args.dataset in ["tictactoe"] and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 9
        args.out_dim = 2
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif args.dataset in ["raisin"] and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 7
        args.out_dim = 2
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif args.dataset in ["phishing_binary"] and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 9
        args.out_dim = 2
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif (args.dataset in ["wifi"] or re.match(r"wifi_corrupt_(.+)", args.dataset)) \
            and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 7
        args.out_dim = 4
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif re.match(r"gaussian_rule_(.+)_regression_10d_(.+)", args.dataset) and args.arch in ["mlp5_sigmoid", "resmlp5_sigmoid"]:
        args.in_dim = 10
        args.out_dim = 1
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "regression"
    elif re.match(r"binary_rule_(.+)_regression_10d_(.+)", args.dataset) and args.arch in ["mlp5_sigmoid", "resmlp5_sigmoid"]:
        args.in_dim = 10
        args.out_dim = 1
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "regression"
    elif re.match(r"zero_one_rule_(.+)_regression_10d_(.+)", args.dataset) \
            and args.arch in ["mlp5_sigmoid", "resmlp5_sigmoid",
                              *[f"mlp{i}" for i in range(2, 11)],
                              *[f"mlp{i}_sigmoid" for i in range(2, 11)],
                              *[f"resmlp{i}_sigmoid" for i in range(2, 11)],]:
        args.in_dim = 10
        args.out_dim = 1
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "regression"
    elif re.match(r"uniform_2_rule_(.+)_regression_10d_(.+)", args.dataset) and args.arch in ["mlp5_sigmoid", "resmlp5_sigmoid"]:
        args.in_dim = 10
        args.out_dim = 1
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "regression"
    elif re.match(r"binary_rule_(.+)_classification_10d_(.+)", args.dataset) and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 10
        args.out_dim = 2
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif re.match(r"uniform_1_rule_(.+)_classification_10d_(.+)", args.dataset) and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 10
        args.out_dim = 2
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif re.match(r"gaussian_rule_(.+)_classification_8d_(.+)", args.dataset) and args.arch in ["mlp5", "resmlp5"]:
        args.in_dim = 8
        args.out_dim = 2
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "classification"
    elif re.match(r"census_rule_(.+)_regression_12d_(.+)", args.dataset) and args.arch in ["mlp5_sigmoid", "resmlp5_sigmoid"]:
        args.in_dim = 12
        args.out_dim = 1
        args.model_kwargs = {"in_dim": args.in_dim, "hidd_dim": 100, "out_dim": args.out_dim}
        args.task = "regression"
    elif args.dataset in ["mnist", "simplemnist"] and args.arch in ["lenet"]:
        args.model_kwargs = {}
        args.task = "classification"
    elif args.dataset in ["mnist", "simplemnist"] and args.arch in ["resnet20", "resnet32", "resnet44", "vgg13_bn", "vgg16_bn"]:
        args.model_kwargs = {"input_channel": 1, "num_classes": 10}
        args.task = "classification"
    elif args.dataset in ["simpleisthree"] and args.arch in ["lenet", "resnet20", "resnet32", "resnet44", "vgg13_bn", "vgg16_bn"]:
        args.model_kwargs = {"input_channel": 1, "num_classes": 1}
        args.task = "logistic_regression"
    elif args.dataset.startswith("celeba") and args.arch in ["alexnet", "vgg13_bn", "vgg16_bn", "resnet18", "resnet34", "resnet50"]:
        args.model_kwargs = {"num_classes": 1}
        args.task = "logistic_regression"
    elif args.dataset in ["dog_bird", "reddog_bluebird", "bg_bird", "redbg_bluebird"] \
            and args.arch in ["alexnet", "vgg13_bn", "vgg16_bn", "resnet18", "resnet34", "resnet50"]:
        args.model_kwargs = {"num_classes": 1}
        args.task = "logistic_regression"
    elif args.dataset in ["shapenet"] and args.arch in ["pointnet", "pointnet2", "pointconv"]:
        args.model_kwargs = {"num_classes": 16}
        args.task = "pointcloud_classification"
    else:
        raise NotImplementedError(f"[Undefined] Dataset: {args.dataset}, Model: {args.arch}")


def setup_train_model_tabular(args):
    args.dataset_model = generate_dataset_model_desc(args)
    _init_model_setting(args)

    set_seed(args.seed)

    # the save folder
    args.save_root = osp.join(args.save_root, args.dataset_model)
    makedirs(args.save_root)
    args.hostname = socket.gethostname()
    save_args(args=args, save_path=osp.join(args.save_root, "hparams.json"))


def setup_train_model_image_tiny(args):
    args.dataset_model = generate_dataset_model_desc(args)
    _init_model_setting(args)

    set_seed(args.seed)

    # the save folder
    args.save_root = osp.join(args.save_root, args.dataset_model)
    makedirs(args.save_root)
    args.hostname = socket.gethostname()
    save_args(args=args, save_path=osp.join(args.save_root, "hparams.json"))


def setup_train_model_image_large(args):
    args.dataset_model = generate_dataset_model_desc(args)
    _init_model_setting(args)

    set_seed(args.seed)

    # the save folder
    args.save_root = osp.join(args.save_root, args.dataset_model)
    makedirs(args.save_root)
    args.hostname = socket.gethostname()
    save_args(args=args, save_path=osp.join(args.save_root, "hparams.json"))


def setup_train_model_pointcloud(args):
    args.dataset_model = generate_dataset_model_desc(args)
    _init_model_setting(args)

    set_seed(args.seed)

    # the save folder
    args.save_root = osp.join(args.save_root, args.dataset_model)
    makedirs(args.save_root)
    args.hostname = socket.gethostname()
    save_args(args=args, save_path=osp.join(args.save_root, "hparams.json"))


def setup_eval_interaction_image(args):
    assert args.model_args is not None
    args.dataset_model = parse_dataset_model_desc(args.model_args)
    args.dataset = args.dataset_model["dataset"]
    args.arch = args.dataset_model["arch"]
    args.seed = args.dataset_model["seed"]
    args.batch_size = args.dataset_model["batch_size"]
    _init_model_setting(args)
    set_seed(args.seed)

    args.sparsified_and_or_desc = generate_sparsified_and_or_desc(args)
    args.save_root = osp.join(args.save_root, args.model_args,
                              f"dim={args.selected_dim}"
                              f"_input={args.input}"
                              f"_baseline={args.baseline}"
                              f"_{args.sparsified_and_or_desc}")


    args.manual_segment_root = osp.join(args.manual_segment_root, f"{args.dataset}-segments")

    makedirs(args.save_root)
    args.hostname = socket.gethostname()
    save_args(args=args, save_path=osp.join(args.save_root, "hparams.json"))


def setup_eval_interaction_pointcloud(args):
    assert args.model_args is not None
    args.dataset_model = parse_dataset_model_desc(args.model_args)
    args.dataset = args.dataset_model["dataset"]
    args.arch = args.dataset_model["arch"]
    args.seed = args.dataset_model["seed"]
    args.batch_size = args.dataset_model["batch_size"]
    _init_model_setting(args)
    set_seed(args.seed)

    args.sparsified_and_or_desc = generate_sparsified_and_or_desc(args)
    args.save_root = osp.join(args.save_root, args.model_args,
                              f"dim={args.selected_dim}"
                              f"_baseline={args.baseline}"
                              f"_{args.sparsified_and_or_desc}")

    args.manual_segment_root = osp.join(args.manual_segment_root, f"{args.dataset}-segments")

    makedirs(args.save_root)
    args.hostname = socket.gethostname()
    save_args(args=args, save_path=osp.join(args.save_root, "hparams.json"))


def setup_eval_interaction_tabular(args):
    assert args.model_args is not None
    args.dataset_model = parse_dataset_model_desc(args.model_args)
    args.dataset = args.dataset_model["dataset"]
    args.arch = args.dataset_model["arch"]
    args.seed = args.dataset_model["seed"]
    args.batch_size = args.dataset_model["batch_size"]
    _init_model_setting(args)
    set_seed(args.seed)

    args.sparsified_and_or_desc = generate_sparsified_and_or_desc(args)
    args.save_root = osp.join(args.save_root, args.model_args,
                              f"dim={args.selected_dim}"
                              f"_baseline={args.baseline}"
                              f"_{args.sparsified_and_or_desc}")


    if args.task == "classification":
        if args.selected_classes is not None:
            args.selected_classes = [int(i) for i in args.selected_classes.strip().split(",")]

    makedirs(args.save_root)
    args.hostname = socket.gethostname()
    save_args(args=args, save_path=osp.join(args.save_root, "hparams.json"))


if __name__ == '__main__':
    import doctest
    doctest.testmod()