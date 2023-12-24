import json
import os
import os.path as osp
from pprint import pprint
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from datasets.get_dataset import get_dataset
import models.pointcloud as models
from harsanyi import AndOrHarsanyi, AndOrHarsanyiSparsifier
from harsanyi.interaction_utils import flatten, get_mask_input_func_pointcloud
from harsanyi.aog_inference import reorganize_and_or_harsanyi
from baseline_values import get_baseline_value
from tools.train import eval_model
from tools.plot_pointcloud import plot_coalition
from setup_exp import setup_eval_interaction_pointcloud


def parse_args():
    parser = argparse.ArgumentParser(description="evaluate the iou (on point-cloud datasets)")
    parser.add_argument('--data-root', default='/data2/lmj/data', type=str,
                        help="root folder for dataset.")
    parser.add_argument("--model-args", type=str, default=None,
                        help="hyper-parameters for the pre-trained model")
    parser.add_argument('--gpu-id', default=0, type=int, help="set the device.")
    parser.add_argument("--model-root", default="../saved-models", type=str,
                        help='the root folder that stores the pre-trained model')
    parser.add_argument("--save-root", default="../saved-interactions", type=str,
                        help='the root folder to save results')
    parser.add_argument("--manual-segment-root", default="../data/player-segments", type=str,
                        help="the root folder for storing the segmentation info")

    parser.add_argument("--baseline", type=str, default="center",
                        help="configuration of the baseline value")
    parser.add_argument("--selected-dim", type=str, default="gt-log-odds",
                        help="use which dimension to compute interactions")

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
    setup_eval_interaction_pointcloud(args)
    return args


def _get_batch(folder):
    """
    get the input image, label, and player segments from the folder
      - in this folder, make sure all samples are with the same class
    :param folder: str, folder
    :return:
    """
    data_batch = {}
    sample_names = filter(lambda x: x.endswith("label.pth"), os.listdir(folder))
    sample_names = [sample_name[:-10] for sample_name in sample_names]
    sample_names = sorted(sample_names)
    for sample_name in sample_names:
        data_batch[sample_name] = {}
        data_batch[sample_name]["pc"] = osp.join(folder, f"{sample_name}_pc.pth")
        data_batch[sample_name]["label"] = osp.join(folder, f"{sample_name}_label.pth")
        data_batch[sample_name]["players"] = osp.join(folder, f"{sample_name}_players.json")
    return data_batch


def _get_player_names(folder):
    with open(osp.join(folder, "player_names.json"), "r") as f:
        player_names = json.load(f)
    return player_names


def get_model(args):
    model_path = osp.join(args.model_root, args.model_args, "model.pt")
    # model_path = osp.join(args.model_root, args.model_args, "model_best.pt")
    print(f"Load model from {model_path}. Map to cuda:{args.gpu_id}")
    model = models.__dict__[args.arch](**args.model_kwargs)
    model = model.to(args.gpu_id)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(f"cuda:{args.gpu_id}")))
    model.eval()
    return model


def _visualize_players(point_cloud, all_players, player_names, save_folder):
    plot_coalition(
        point_cloud=point_cloud, coalition=all_players,
        save_folder=save_folder, save_name="all_players",
        player_names=player_names
    )
    for i, player in enumerate(all_players):
        plot_coalition(
            point_cloud=point_cloud, coalition=[flatten([player])],
            save_folder=osp.join(save_folder, "vis"), save_name=f"player_{i}",
            player_names=player_names[i:i+1]
        )




def evaluate_single(
        forward_func, selected_dim,
        point_cloud, baseline, label,
        all_players, sparsify_kwargs,
        save_folder
):
    bs, c, n_points = point_cloud.shape
    assert bs == 1
    assert c == 3

    mask_input_fn = get_mask_input_func_pointcloud()
    foreground = list(flatten(all_players))
    indices = np.ones(n_points, dtype=bool)
    indices[foreground] = False
    background = np.arange(n_points)[indices].tolist()
    attributes = [r"$x_{" + str(i) + r"}$" for i in range(len(all_players))]
    logfile_path = osp.join(save_folder, "log.txt")

    # 1. calculate interaction
    calculator = AndOrHarsanyi(
        model=forward_func, selected_dim=selected_dim,
        x=point_cloud, baseline=baseline, y=label,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=32, verbose=0  # calc_bs=32 to avoid out of memory issue
    )
    with torch.no_grad():
        calculator.attribute()
        masks = calculator.get_masks()
        I_and_, I_or_ = calculator.get_interaction()
        calculator.save(save_folder=osp.join(save_folder, "before_sparsify"))

    with open(logfile_path, 'w') as f:  # 检查 efficiency 性质
        f.write(f"[Reward ({selected_dim})]\n")
        f.write(f"v(empty)={calculator.v_empty}\n")
        f.write(f"v(full)={calculator.v_N}\n")
        f.write("\n[Before Sparsifying]\n")
        f.write("sum of I^and:\n")
        f.write(f"\t{I_and_.sum()}\n")
        f.write("sum of I^or:\n")
        f.write(f"\t{I_or_.sum()}\n")
        f.write(f"Sum of I^and and I^or:\n"
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
    with open(logfile_path, 'a') as f:  # 检查 efficiency 性质
        f.write("\n[After Sparsifying]\n")
        f.write(f"\tSum of I^and and I^or: {torch.sum(I_and) + torch.sum(I_or)}\n")


    return torch.cat([I_and, I_or])


if __name__ == '__main__':
    args = parse_args()

    # =========================================
    #     initialize the model and dataset
    # =========================================
    model = get_model(args)
    model_type = model.__class__.__name__
    if model_type == "PointNet":
        forward_func = lambda x: model(x)[0]
    else:
        forward_func = model
    dataset = get_dataset(args.data_root, args.dataset)
    train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size)

    # =========================================
    #    validate the pre-trained model
    # =========================================
    # train_eval_dict = eval_model(model, train_loader, task=args.task)
    # print("train loss:", train_eval_dict)
    test_eval_dict = eval_model(model, test_loader, task=args.task)
    print("test loss:", test_eval_dict)

    # =========================================
    #     evaluate sparsified interaction
    # =========================================

    for class_id in sorted(os.listdir(args.manual_segment_root)):

        data_batch = _get_batch(osp.join(args.manual_segment_root, class_id))
        player_names = _get_player_names(osp.join(args.manual_segment_root, class_id))
        print(f"Class id: {class_id}, # samples: {len(data_batch)}")

        for sample_id in data_batch.keys():
            save_folder = osp.join(args.save_root, class_id, sample_id)
            os.makedirs(save_folder, exist_ok=True)

            # load input sample, label, baseline, player segments
            point_cloud = torch.load(osp.join(data_batch[sample_id]["pc"]), map_location=torch.device("cpu"))
            label = torch.load(osp.join(data_batch[sample_id]["label"]), map_location=torch.device("cpu"))
            baseline = get_baseline_value(point_cloud, baseline_config=args.baseline)
            with open(data_batch[sample_id]["players"], "r") as f:
                all_players = json.load(f)

            print("sample", sample_id, "| # of players", len(all_players), "| pc shape", point_cloud.shape)

            # visualize the players
            _visualize_players(point_cloud, all_players, player_names, save_folder)

            point_cloud = point_cloud.to(args.gpu_id)
            label = label.item()
            baseline = baseline.to(args.gpu_id)

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
                point_cloud=point_cloud, baseline=baseline, label=label,
                all_players=all_players, save_folder=save_folder,
                sparsify_kwargs=sparsify_kwargs,
            )




