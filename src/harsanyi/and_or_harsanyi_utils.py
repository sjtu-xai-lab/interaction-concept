import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import os
import os.path as osp

from typing import Callable, List, Tuple, Union, Dict
import sys

from .interaction_utils import generate_all_masks, generate_subset_masks, generate_reverse_subset_masks, \
    generate_set_with_intersection_masks


def get_reward2Iand_mat(dim):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to and-interaction (Harsanyi)
    :param dim: the input dimension n
    :return: a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    # for i in tqdm(range(n_masks), ncols=100, desc="Generating mask"):
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ===============================================================================================
        # Note: I(S) = \sum_{L\subseteq S} (-1)^{s-l} v(L)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        L_indices = (L_indices == True).nonzero(as_tuple=False)
        assert mask_Ls.shape[0] == L_indices.shape[0]
        row[L_indices] = torch.pow(-1., mask_S.sum() - mask_Ls.sum(dim=1)).unsqueeze(1)
        # ===============================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_reward2Ior_mat(dim):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to or-interaction
    :param dim: the input dimension n
    :return: a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ===============================================================================================
        # Note: I(S) = -\sum_{L\subseteq S} (-1)^{s+(n-l)-n} v(N\L) if S is not empty
        if mask_S.sum() == 0:
            row[i] = 1.
        else:
            mask_NLs, NL_indices = generate_reverse_subset_masks(mask_S, all_masks)
            NL_indices = (NL_indices == True).nonzero(as_tuple=False)
            assert mask_NLs.shape[0] == NL_indices.shape[0]
            row[NL_indices] = - torch.pow(-1., mask_S.sum() + mask_NLs.sum(dim=1) + dim).unsqueeze(1)
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_Iand2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = \sum_{L\subseteq S} I(S)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


def get_Ior2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    mask_empty = torch.zeros(dim).bool()
    _, empty_indice = generate_subset_masks(mask_empty, all_masks)
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = I(\emptyset) + \sum_{L: L\union S\neq \emptyset} I(S)
        row[empty_indice] = 1.
        mask_Ls, L_indices = generate_set_with_intersection_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


# ==============================================
#     For sparsifying and-or interactions
# ==============================================
def l1_on_given_dim(vector: torch.Tensor, indices: List) -> torch.Tensor:
    assert len(vector.shape) == 1
    strength = torch.abs(vector)
    return torch.sum(strength[indices])


def generate_ckpt_id_list(niter: int, nckpt: int) -> List:
    ckpt_id_list = list(range(niter))[::max(1, niter // nckpt)]
    # force the last iteration to be a checkpoint
    if niter - 1 not in ckpt_id_list:
        ckpt_id_list.append(niter - 1)
    return ckpt_id_list


def _train_p(
        rewards: torch.Tensor,
        loss_type: str,
        lr: float,
        niter: int,
        reward2Iand: torch.Tensor = None,
        reward2Ior: torch.Tensor = None
) -> Tuple[torch.Tensor, Dict, Dict]:
    device = rewards.device
    n_dim = int(np.log2(rewards.numel()))
    if reward2Iand is None:
        reward2Iand = get_reward2Iand_mat(n_dim).to(device)
    if reward2Ior is None:
        reward2Ior = get_reward2Ior_mat(n_dim).to(device)

    p = torch.zeros_like(rewards).requires_grad_(True)
    optimizer = optim.SGD([p], lr=0.0, momentum=0.9)

    log_lr = np.log10(lr)
    eta_list = np.logspace(log_lr, log_lr - 1, niter)

    if loss_type == "l1":
        losses = {"loss": []}
    elif loss_type.startswith("l1_on"):
        ratio = float(loss_type.split("_")[-1])
        Iand_p = torch.matmul(reward2Iand, 0.5 * rewards + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * rewards - p)
        num_noisy_pattern = int(ratio * (Iand_p.shape[0] + Ior_p.shape[0]))
        print("# noisy patterns", num_noisy_pattern)
        noisy_pattern_indices = torch.argsort(torch.abs(torch.cat([Iand_p, Ior_p]))).tolist()[:num_noisy_pattern]
        losses = {"loss": [], "noise_ratio": []}
    else:
        raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

    progresses = {"I_and": [], "I_or": []}
    ckpt_id_list = generate_ckpt_id_list(niter=niter, nckpt=20)

    pbar = tqdm(range(niter), desc="Optimizing p", ncols=100)
    for it in pbar:
        Iand_p = torch.matmul(reward2Iand, 0.5 * rewards + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * rewards - p)

        if loss_type == "l1":
            loss = torch.sum(torch.abs(Iand_p)) + torch.sum(torch.abs(Ior_p))  # 02-27: L1 penalty.
            losses["loss"].append(loss.item())
        elif loss_type.startswith("l1_on"):
            loss = l1_on_given_dim(torch.cat([Iand_p, Ior_p]), indices=noisy_pattern_indices)
            losses["loss"].append(loss.item())
            losses["noise_ratio"].append(loss.item() / torch.sum(torch.abs(torch.cat([Iand_p, Ior_p]))).item())
        else:
            raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

        if it + 1 < niter:
            optimizer.zero_grad()
            optimizer.param_groups[0]["lr"] = eta_list[it]
            loss.backward()
            optimizer.step()

        if it in ckpt_id_list:
            progresses["I_and"].append(Iand_p.detach().cpu().numpy())
            progresses["I_or"].append(Ior_p.detach().cpu().numpy())
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

    return p.detach(), losses, progresses


def _train_p_q(
        rewards: torch.Tensor,
        loss_type: str,
        lr: float,
        niter: int,
        qbound: Union[float, torch.Tensor],
        reward2Iand: torch.Tensor = None,
        reward2Ior: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
    device = rewards.device
    n_dim = int(np.log2(rewards.numel()))
    if reward2Iand is None:
        reward2Iand = get_reward2Iand_mat(n_dim).to(device)
    if reward2Ior is None:
        reward2Ior = get_reward2Ior_mat(n_dim).to(device)

    log_lr = np.log10(lr)
    eta_list = np.logspace(log_lr, log_lr - 1, niter)

    p = torch.zeros_like(rewards).requires_grad_(True)
    q = torch.zeros_like(rewards).requires_grad_(True)
    optimizer = optim.SGD([p, q], lr=0.0, momentum=0.9)

    if loss_type == "l1":
        losses = {"loss": []}
    elif loss_type.startswith("l1_on"):
        ratio = float(loss_type.split("_")[-1])
        Iand_p = torch.matmul(reward2Iand, 0.5 * rewards + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * rewards - p)
        num_noisy_pattern = int(ratio * (Iand_p.shape[0] + Ior_p.shape[0]))
        print("# noisy patterns", num_noisy_pattern)
        noisy_pattern_indices = torch.argsort(torch.abs(torch.cat([Iand_p, Ior_p]))).tolist()[:num_noisy_pattern]
        losses = {"loss": [], "noise_ratio": []}
    else:
        raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

    progresses = {"I_and": [], "I_or": []}
    ckpt_id_list = generate_ckpt_id_list(niter=niter, nckpt=20)

    pbar = tqdm(range(niter), desc="Optimizing pq", ncols=100)
    for it in pbar:
        # # The case when min/max are tensors: not supported until torch 1.9.1
        # q.data = torch.clamp(q.data, -qbound, qbound)
        q.data = torch.max(torch.min(q.data, qbound), -qbound)
        Iand_p = torch.matmul(reward2Iand, 0.5 * (rewards + q) + p)
        Ior_p = torch.matmul(reward2Ior, 0.5 * (rewards + q) - p)

        if loss_type == "l1":
            loss = torch.sum(torch.abs(Iand_p)) + torch.sum(torch.abs(Ior_p))  # 02-27: L1 penalty.
            losses["loss"].append(loss.item())
        elif loss_type.startswith("l1_on"):
            loss = l1_on_given_dim(torch.cat([Iand_p, Ior_p]), indices=noisy_pattern_indices)
            losses["loss"].append(loss.item())
            losses["noise_ratio"].append(loss.item() / torch.sum(torch.abs(torch.cat([Iand_p, Ior_p]))).item())
        else:
            raise NotImplementedError(f"Loss type {loss_type} unrecognized.")

        if it + 1 < niter:
            optimizer.zero_grad()
            optimizer.param_groups[0]["lr"] = eta_list[it]
            loss.backward()
            optimizer.step()

        if it in ckpt_id_list:
            progresses["I_and"].append(Iand_p.detach().cpu().numpy())
            progresses["I_or"].append(Ior_p.detach().cpu().numpy())
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

    return p.detach(), q.detach(), losses, progresses

