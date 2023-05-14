import torch
import numpy as np
from typing import Callable, Tuple, List
from tqdm import tqdm
from .and_or_harsanyi_utils import get_Iand2reward_mat, get_Ior2reward_mat, get_reward2Iand_mat, get_reward2Ior_mat


def get_and_or_harsanyi_inference_func(
        all_masks: torch.Tensor,
        I_and: torch.Tensor,
        I_or: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    assert len(all_masks.shape) == 2

    empty_indices = torch.all(torch.logical_not(all_masks), dim=1)
    assert empty_indices.sum().item() == 1

    def inference_func(input_mask: torch.Tensor) -> torch.Tensor:
        assert all_masks.shape[1] == input_mask.shape[0]

        if torch.any(input_mask):
            act_indices_and = torch.all(torch.logical_or(torch.logical_not(all_masks), input_mask[None, :]), dim=1)
            act_indices_or = torch.any(all_masks[:, input_mask], dim=1)
            act_indices_and = torch.logical_or(empty_indices, act_indices_and)
            act_indices_or = torch.logical_or(empty_indices, act_indices_or)
        else:
            act_indices_and = empty_indices.clone()
            act_indices_or = empty_indices.clone()

        return I_and[act_indices_and].sum() + I_or[act_indices_or].sum()

    return inference_func


def reorganize_and_or_harsanyi(
        all_masks: torch.Tensor,
        I_and: torch.Tensor,
        I_or: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    To combine the single-order, zero-order components in and-or interactions together
    :param all_masks:
    :param I_and:
    :param I_or:
    :return:
    """
    I_and_ = I_and.clone()
    I_or_ = I_or.clone()

    comb_indices = torch.sum(all_masks, dim=1) <= 1

    I_and_[comb_indices] = I_and_[comb_indices] + I_or_[comb_indices]
    I_or_[comb_indices] = 0

    return I_and_, I_or_


def remove_noisy_and_or_harsanyi(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        threshold: float  # AOG 中的 interaction 强度占全部 interaction 的强度
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    """

    :param I_and:
    :param I_or:
    :param threshold: the threshold for total strength of removed interactions
    :return:
    """
    if threshold == 0:
        I_and_retained_indices = list(np.arange(I_and.shape[0]).tolist())
        I_or_retained_indices = list(np.arange(I_or.shape[0]).tolist())
        return I_and, I_or, I_and_retained_indices, I_or_retained_indices

    interactions = torch.cat([I_and, I_or]).clone()
    total_strength = torch.abs(interactions).sum() + 1e-7
    strength_order = torch.argsort(torch.abs(interactions))

    removed_ratio = torch.cumsum(torch.abs(interactions[strength_order]), dim=0) / total_strength
    first_retain_id = (removed_ratio > threshold).nonzero()[0, 0]
    removed_indices = strength_order[:first_retain_id]
    retained_indices = strength_order[first_retain_id:]

    interactions[removed_indices] = 0  # set the interaction of removed patterns to zero

    I_and_retained_indices = retained_indices[retained_indices < I_and.shape[0]].tolist()
    I_or_retained_indices = retained_indices[retained_indices >= I_and.shape[0]] - I_and.shape[0]
    I_or_retained_indices = I_or_retained_indices.tolist()

    I_and_ = interactions[:I_and.shape[0]]
    I_or_ = interactions[I_and.shape[0]:]

    # return I_and_, I_or_, first_retain_id
    return I_and_, I_or_, I_and_retained_indices, I_or_retained_indices


def generate_and_or_harsanyi_remove_order_min_unfaith(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        n_greedy: int = 40  # 每次考虑 n_greedy 个 interaction pattern 作为 candidate，从中删去一个
) -> np.ndarray:
    device = I_and.device
    n_players = int(np.log2(I_and.shape[0]))
    Iand2reward = get_Iand2reward_mat(n_players).to(device)
    Ior2reward = get_Ior2reward_mat(n_players).to(device)

    rewards = Iand2reward @ I_and + Ior2reward @ I_or

    interactions = torch.cat([I_and, I_or]).clone()
    strength_order = torch.argsort(torch.abs(interactions)).tolist()  # from low-strength to high-strength
    remove_order = []

    for n_remove in tqdm(range(interactions.shape[0]), desc="calc remove order", ncols=100):
        candidates = strength_order[:n_greedy]
        to_remove = candidates[0]
        interactions_ = interactions.clone()
        interactions_[to_remove] = 0.
        I_and_ = interactions_[:I_and.shape[0]]
        I_or_ = interactions_[I_and.shape[0]:]
        rewards_aog_ = Iand2reward @ I_and_ + Ior2reward @ I_or_
        unfaith = torch.sum(torch.square(rewards - rewards_aog_))

        for i in range(1, len(candidates)):
            candidate = candidates[i]
            interactions_ = interactions.clone()
            interactions_[candidate] = 0.
            I_and_ = interactions_[:I_and.shape[0]]
            I_or_ = interactions_[I_and.shape[0]:]
            rewards_aog_ = Iand2reward @ I_and_ + Ior2reward @ I_or_
            unfaith_ = torch.sum(torch.square(rewards - rewards_aog_))
            if unfaith_ < unfaith:
                to_remove = candidate
                unfaith = unfaith_

        interactions[to_remove] = 0.
        remove_order.append(to_remove)
        strength_order.remove(to_remove)

    return np.array(remove_order)


def remove_noisy_and_or_harsanyi_given_remove_order(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        remove_order: np.ndarray,
        threshold: float = None,  # 删去的 interaction 强度占全部 interaction 的强度
        retain_num: int = None,  # 最终留下来的 interaction pattern 数量
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    if threshold is not None:
        return _remove_noisy_given_order_remove_ratio(I_and=I_and, I_or=I_or,
                                                      remove_order=remove_order, threshold=threshold)
    if retain_num is not None:
        return _remove_noisy_given_order_retain_num(I_and=I_and, I_or=I_or,
                                                    remove_order=remove_order, retain_num=retain_num)


def _remove_noisy_given_order_remove_ratio(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        remove_order: np.ndarray,
        threshold: float,  # 删去的 interaction 强度占全部 interaction 的强度
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    assert threshold < 1

    if threshold == 0:
        I_and_retained_indices = list(np.arange(I_and.shape[0]).tolist())
        I_or_retained_indices = list(np.arange(I_or.shape[0]).tolist())
        return I_and, I_or, I_and_retained_indices, I_or_retained_indices

    interactions = torch.cat([I_and, I_or]).clone()
    total_strength = torch.abs(interactions).sum() + 1e-7

    removed_ratio = torch.cumsum(torch.abs(interactions[remove_order]), dim=0) / total_strength
    first_retain_id = (removed_ratio > threshold).nonzero()[0, 0]
    removed_indices = remove_order[:first_retain_id]
    retained_indices = remove_order[first_retain_id:]

    interactions[removed_indices] = 0  # set the interaction of removed patterns to zero

    I_and_retained_indices = retained_indices[retained_indices < I_and.shape[0]].tolist()
    I_or_retained_indices = retained_indices[retained_indices >= I_and.shape[0]] - I_and.shape[0]
    I_or_retained_indices = I_or_retained_indices.tolist()

    I_and_ = interactions[:I_and.shape[0]]
    I_or_ = interactions[I_and.shape[0]:]

    # return I_and_, I_or_, first_retain_id
    return I_and_, I_or_, I_and_retained_indices, I_or_retained_indices


def _remove_noisy_given_order_retain_num(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        remove_order: np.ndarray,
        retain_num: int = None,  # 最终留下来的 interaction pattern 数量
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:

    assert retain_num > 0

    interactions = torch.cat([I_and, I_or]).clone()
    first_retain_id = interactions.shape[0] - retain_num
    removed_indices = remove_order[:first_retain_id]
    retained_indices = remove_order[first_retain_id:]
    assert len(retained_indices) == retain_num

    interactions[removed_indices] = 0  # set the interaction of removed patterns to zero

    I_and_retained_indices = retained_indices[retained_indices < I_and.shape[0]].tolist()
    I_or_retained_indices = retained_indices[retained_indices >= I_and.shape[0]] - I_and.shape[0]
    I_or_retained_indices = I_or_retained_indices.tolist()

    I_and_ = interactions[:I_and.shape[0]]
    I_or_ = interactions[I_and.shape[0]:]

    # return I_and_, I_or_, first_retain_id
    return I_and_, I_or_, I_and_retained_indices, I_or_retained_indices


def remove_noisy_and_or_harsanyi_min_unfaith(
        I_and: torch.Tensor,
        I_or: torch.Tensor,
        threshold: float,  # 删去的 interaction 强度占全部 interaction 的强度
        n_greedy: int = 40  # 每次考虑 n_greedy 个 interaction pattern 作为 candidate，从中删去一个
) -> Tuple[torch.Tensor, torch.Tensor, List, List]:

    if threshold == 0:
        I_and_retained_indices = list(np.arange(I_and.shape[0]).tolist())
        I_or_retained_indices = list(np.arange(I_or.shape[0]).tolist())
        return I_and, I_or, I_and_retained_indices, I_or_retained_indices

    device = I_and.device
    n_players = int(np.log2(I_and.shape[0]))
    Iand2reward = get_Iand2reward_mat(n_players).to(device)
    Ior2reward = get_Ior2reward_mat(n_players).to(device)

    rewards = Iand2reward @ I_and + Ior2reward @ I_or

    interactions = torch.cat([I_and, I_or]).clone()
    interactions_original = interactions.clone()
    strength_order = torch.argsort(torch.abs(interactions)).tolist()  # from low-strength to high-strength
    removed_indices = []

    for n_remove in tqdm(range(interactions.shape[0]), desc="removing", ncols=100):
        candidates = strength_order[:n_greedy]
        to_remove = candidates[0]
        interactions_ = interactions.clone()
        interactions_[to_remove] = 0.
        I_and_ = interactions_[:I_and.shape[0]]
        I_or_ = interactions_[I_and.shape[0]:]
        rewards_aog_ = Iand2reward @ I_and_ + Ior2reward @ I_or_
        unfaith = torch.sum(torch.square(rewards - rewards_aog_))

        for i in range(1, len(candidates)):
            candidate = candidates[i]
            interactions_ = interactions.clone()
            interactions_[candidate] = 0.
            I_and_ = interactions_[:I_and.shape[0]]
            I_or_ = interactions_[I_and.shape[0]:]
            rewards_aog_ = Iand2reward @ I_and_ + Ior2reward @ I_or_
            unfaith_ = torch.sum(torch.square(rewards - rewards_aog_))
            if unfaith_ < unfaith:
                to_remove = candidate
                unfaith = unfaith_

        interactions[to_remove] = 0.
        ratio = 1 - torch.sum(torch.abs(interactions)) / (torch.sum(torch.abs(interactions_original)) + 1e-7)
        if ratio > threshold:
            interactions[to_remove] = interactions_original[to_remove]
            break
        removed_indices.append(to_remove)
        strength_order.remove(to_remove)

    I_and_ = interactions[:I_and.shape[0]]
    I_or_ = interactions[I_and.shape[0]:]
    retained_indices = np.array([i for i in range(interactions.shape[0]) if i not in removed_indices])

    I_and_retained_indices = retained_indices[retained_indices < I_and.shape[0]].tolist()
    I_or_retained_indices = retained_indices[retained_indices >= I_and.shape[0]] - I_and.shape[0]
    I_or_retained_indices = I_or_retained_indices.tolist()

    return I_and_, I_or_, I_and_retained_indices, I_or_retained_indices


if __name__ == '__main__':
    all_masks = torch.Tensor([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]).bool()
    I_and = torch.randn(8)
    I_or = torch.randn(8)

    inference_func = get_and_or_harsanyi_inference_func(all_masks, I_and, I_or)

    print(inference_func(torch.Tensor([0, 1, 1]).bool()),
          I_and[[0, 1, 2, 3]].sum() + I_or[[0, 1, 2, 3, 5, 6, 7]].sum())

    print(inference_func(torch.Tensor([0, 0, 0]).bool()),
          I_and[0] + I_or[0])

    print(inference_func(torch.Tensor([1, 1, 1]).bool()),
          I_and.sum() + I_or.sum())

    I_and_, I_or_ = reorganize_and_or_harsanyi(all_masks, I_and, I_or)

    print("Before reorganization")
    print(I_and)
    print(I_or)
    print("After reorganization")
    print(I_and_)
    print(I_or_)
