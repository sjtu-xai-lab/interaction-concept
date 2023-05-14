import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn

from typing import Union, Iterable, List, Tuple, Callable

from .and_or_harsanyi_utils import get_reward2Iand_mat, get_reward2Ior_mat, _train_p, _train_p_q
from .interaction_utils import calculate_all_subset_outputs, calculate_output_empty, calculate_output_N, get_reward
from .plot import plot_simple_line_chart, plot_interaction_progress


class AndHarsanyi(object):
    def __init__(
            self,
            model: Union[nn.Module, Callable],
            selected_dim: Union[None, str],
            x: torch.Tensor,
            baseline: torch.Tensor,
            y: Union[torch.Tensor, int, None],
            all_players: Union[None, tuple, list] = None,
            background: Union[None, tuple, list] = None,
            mask_input_fn: Callable = None,
            calc_bs: int = None,
            verbose: int = 1,
    ):
        assert x.shape[0] == baseline.shape[0] == 1
        self.model = model
        self.selected_dim = selected_dim
        self.input = x
        self.target = y
        self.baseline = baseline
        self.device = x.device
        self.verbose = verbose

        self.all_players = all_players  # customize players
        if background is None:
            background = []
        self.background = background  # players that always exists (default: emptyset [])

        self.mask_input_fn = mask_input_fn  # for image data
        self.calc_bs = calc_bs

        if all_players is not None:  # image data
            self.n_players = len(all_players)
        else:
            self.n_players = self.input.shape[1]

        if self.verbose == 1:
            print("[AndHarsanyi] Generating v->I^and matrix:")
        self.reward2Iand = get_reward2Iand_mat(self.n_players).to(self.device)
        if self.verbose == 1:
            print("[AndHarsanyi] Finish.")

        # calculate v(N) and v(empty)
        with torch.no_grad():
            self.output_empty = calculate_output_empty(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, verbose=self.verbose
            )
            self.output_N = calculate_output_N(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, verbose=self.verbose
            )
            # self.output_empty = model(self.baseline)
            # self.output_N = model(self.input)
        if self.selected_dim.endswith("-v0"):
            self.v0 = get_reward(self.output_empty, self.selected_dim[:-3], gt=y)
        else:
            self.v0 = 0
        self.v_N = get_reward(self.output_N, self.selected_dim, gt=y, v0=self.v0)
        self.v_empty = get_reward(self.output_empty, self.selected_dim, gt=y, v0=self.v0)

    def attribute(self):
        with torch.no_grad():
            self.masks, outputs = calculate_all_subset_outputs(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, calc_bs=self.calc_bs,
                verbose=self.verbose
            )
        self.rewards = get_reward(outputs, self.selected_dim, gt=self.target, v0=self.v0)
        self.Iand = torch.matmul(self.reward2Iand, self.rewards)

    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        masks = self.masks.cpu().numpy()
        rewards = self.rewards.cpu().numpy()
        sample = self.input.cpu().numpy()
        np.save(osp.join(save_folder, "rewards.npy"), rewards)
        np.save(osp.join(save_folder, "masks.npy"), masks)
        np.save(osp.join(save_folder, "sample.npy"), sample)
        Iand = self.Iand.cpu().numpy()
        np.save(osp.join(save_folder, "Iand.npy"), Iand)

    def get_interaction(self):
        return self.Iand

    def get_masks(self):
        return self.masks


class AndOrHarsanyi(object):
    def __init__(
            self,
            model: Union[nn.Module, Callable],
            selected_dim: Union[None, str],
            x: torch.Tensor,
            baseline: torch.Tensor,
            y: Union[torch.Tensor, int, None],
            all_players: Union[None, tuple, list] = None,
            background: Union[None, tuple, list] = None,
            mask_input_fn: Callable = None,
            calc_bs: int = None,
            verbose: int = 1
    ):
        assert x.shape[0] == baseline.shape[0] == 1
        self.model = model
        self.selected_dim = selected_dim
        self.input = x
        self.target = y
        self.baseline = baseline
        self.device = x.device
        self.verbose = verbose

        self.all_players = all_players  # customize players
        if background is None:
            background = []
        self.background = background  # players that always exists (default: emptyset [])

        self.mask_input_fn = mask_input_fn  # for image data
        self.calc_bs = calc_bs

        if all_players is not None:  # image data
            self.n_players = len(all_players)
        else:
            self.n_players = self.input.shape[1]

        if self.verbose == 1:
            print("[AndOrHarsanyi] Generating v->I^and matrix:")
        self.reward2Iand = get_reward2Iand_mat(self.n_players).to(self.device)
        if self.verbose == 1:
            print("[AndOrHarsanyi] Generating v->I^or matrix:")
        self.reward2Ior = get_reward2Ior_mat(self.n_players).to(self.device)
        if self.verbose == 1:
            print("[AndOrHarsanyi] Finish.")

        # calculate v(N) and v(empty)
        with torch.no_grad():
            self.output_empty = calculate_output_empty(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, verbose=self.verbose
            )
            self.output_N = calculate_output_N(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, verbose=self.verbose
            )

        if self.selected_dim.endswith("-v0"):
            self.v0 = get_reward(self.output_empty, self.selected_dim[:-3], gt=y)
        else:
            self.v0 = 0

        self.v_N = get_reward(self.output_N, self.selected_dim, gt=y, v0=self.v0)
        self.v_empty = get_reward(self.output_empty, self.selected_dim, gt=y, v0=self.v0)

    def attribute(self):
        with torch.no_grad():
            self.masks, outputs = calculate_all_subset_outputs(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, calc_bs=self.calc_bs,
                verbose=self.verbose
            )
        self.rewards = get_reward(outputs, self.selected_dim, gt=self.target, v0=self.v0)
        self.Iand = torch.matmul(self.reward2Iand, self.rewards)
        self.Ior = torch.matmul(self.reward2Ior, self.rewards)

    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        masks = self.masks.cpu().numpy()
        rewards = self.rewards.cpu().numpy()
        sample = self.input.cpu().numpy()
        np.save(osp.join(save_folder, "rewards.npy"), rewards)
        np.save(osp.join(save_folder, "masks.npy"), masks)
        np.save(osp.join(save_folder, "sample.npy"), sample)
        Iand = self.Iand.cpu().numpy()
        Ior = self.Ior.cpu().numpy()
        np.save(osp.join(save_folder, "Iand.npy"), Iand)
        np.save(osp.join(save_folder, "Ior.npy"), Ior)

    def get_interaction(self):
        return 0.5 * self.Iand, 0.5 * self.Ior

    def get_masks(self):
        return self.masks

    def get_and_interaction(self):
        return self.Iand

    def get_or_interaction(self):
        return self.Ior


class AndOrHarsanyiSparsifier(object):
    def __init__(
            self,
            calculator: AndOrHarsanyi,
            trick: str,
            loss: str,
            lr: float,
            niter: int,
            qthres: float = None,
            qstd: str = None,
    ):
        self.calculator = calculator
        self.trick = trick
        self.loss = loss
        self.qthres = qthres
        self.qstd = qstd
        self.lr = lr
        self.niter = niter

        self.p = None
        self.q = None
        self.q_bound = None

    def _init_q_bound(self):
        self.standard = None
        if self.qstd == "none":
            self.q_bound = self.qthres
            return

        if self.qstd == "vS":
            standard = self.calculator.rewards.clone()
        elif self.qstd == "vS-v0":
            standard = self.calculator.rewards - self.calculator.v_empty
        elif self.qstd == "vN":
            standard = self.calculator.v_N.clone()
        elif self.qstd == "vN-v0":
            standard = self.calculator.v_N - self.calculator.v_empty
        elif self.qstd == "maxvS":
            standard, _ = torch.max(torch.abs(self.calculator.rewards), dim=0)
        elif self.qstd == "maxvS-v0":
            standard = torch.max(torch.abs(self.calculator.rewards - self.calculator.v_empty), dim=0)[0]
        else:
            raise NotImplementedError(f"Invalid standard value of `q`: {self.qstd}")

        self.standard = torch.abs(standard)
        self.q_bound = self.qthres * self.standard

        return

    def sparsify(self, verbose_folder=None):
        if self.trick == "p":
            p, losses, progresses = _train_p(
                rewards=self.calculator.rewards,
                loss_type=self.loss,
                lr=self.lr, niter=self.niter,
                reward2Iand=self.calculator.reward2Iand,
                reward2Ior=self.calculator.reward2Ior
            )
            self.p = p.clone()
        elif self.trick == "pq":
            self._init_q_bound()
            p, q, losses, progresses = _train_p_q(
                rewards=self.calculator.rewards,
                loss_type=self.loss,
                lr=self.lr, niter=self.niter,
                qbound=self.q_bound,
                reward2Iand=self.calculator.reward2Iand,
                reward2Ior=self.calculator.reward2Ior
            )
            self.p = p.clone()
            self.q = q.clone()
        else:
            raise NotImplementedError(f"Invalid trick: {self.trick}")

        self._calculate_interaction()

        if verbose_folder is None:
            return

        for k in losses.keys():
            plot_simple_line_chart(
                data=losses[k], xlabel="iteration", ylabel=f"{k}", title="",
                save_folder=verbose_folder, save_name=f"{k}_curve_optimize_{self.trick}"
            )
        for k in progresses.keys():
            plot_interaction_progress(
                interaction=progresses[k], save_path=osp.join(verbose_folder, f"{k}_progress_optimize_{self.trick}.png"),
                order_cfg="descending", title=f"{k} progress during optimization"
            )

        with open(osp.join(verbose_folder, "log.txt"), "w") as f:
            f.write(f"trick: {self.trick} | loss: {self.loss} | lr: {self.lr} | niter: {self.niter}\n")
            f.write(f"for [q] -- threshold: {self.qthres} | standard: {self.qstd}\n")
            if self.q_bound is not None and self.q_bound.numel() < 20:
                f.write(f"\t[q] bound: {self.q_bound}")
            f.write(f"\tThe value of v(N): {self.calculator.v_N}\n")
            f.write(f"\tSum of I^and and I^or: {torch.sum(self.Iand) + torch.sum(self.Ior)}\n")
            f.write(f"\tSum of I^and: {torch.sum(self.Iand)}\n")
            f.write(f"\tSum of I^or: {torch.sum(self.Ior)}\n")
            f.write(f"\t|I^and|+|I^or|: {torch.sum(torch.abs(self.Iand)) + torch.sum(torch.abs(self.Ior)).item()}\n")
            f.write("\tDuring optimizing,\n")
            for k, v in losses.items():
                f.write(f"\t\t{k}: {v[0]} -> {v[-1]}\n")

    def _calculate_interaction(self):
        rewards = self.calculator.rewards
        if self.trick == "p":
            self.Iand = torch.matmul(self.calculator.reward2Iand, 0.5 * rewards + self.p).detach()
            self.Ior = torch.matmul(self.calculator.reward2Ior, 0.5 * rewards - self.p).detach()
        elif self.trick == "pq":
            self.Iand = torch.matmul(self.calculator.reward2Iand, 0.5 * (rewards + self.q) + self.p).detach()
            self.Ior = torch.matmul(self.calculator.reward2Ior, 0.5 * (rewards + self.q) - self.p).detach()
        else:
            raise NotImplementedError(f"Invalid trick: {self.trick}")

    def save(self, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        Iand = self.Iand.cpu().numpy()
        Ior = self.Ior.cpu().numpy()
        np.save(osp.join(save_folder, "Iand.npy"), Iand)
        np.save(osp.join(save_folder, "Ior.npy"), Ior)
        p = self.p.cpu().numpy()
        np.save(osp.join(save_folder, "p.npy"), p)
        if self.q is not None:
            q = self.q.cpu().numpy()
            np.save(osp.join(save_folder, "q.npy"), q)

    def get_interaction(self):
        return self.Iand, self.Ior

    def get_masks(self):
        return self.calculator.masks
