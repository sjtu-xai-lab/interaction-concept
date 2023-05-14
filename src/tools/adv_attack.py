import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Callable, Union, Tuple, List, Dict


def pgd_attack(
        model: Union[nn.Module, Callable],
        x: torch.Tensor,
        y: Union[torch.Tensor, int, float],
        loss_func: Callable,
        step_size: float,
        epsilon: Union[float, torch.Tensor],
        n_step: int,
        distance: str = "l_inf",
        bound_max: Union[None, torch.Tensor, float] = None,
        bound_min: Union[None, torch.Tensor, float] = None
) -> torch.Tensor:
    """
    Perform PGD attack [1] on the input sample, and return the adversarial sample.
      In each step of the PGD attack, perform the following operation,

            max_{x'}     loss(model(x'), y)
              s.t.    distance(x',x) <= epsilon,
                      x' still lies in some range

      In PGD attack, usually we iteratively optimize the above equation via gradient ascent.

    [1] Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks."
        arXiv preprint arXiv:1706.06083 (2017).

    :param model: The model or the forward function
    :param x: The input sample on which we want to perform an attack
    :param y: The ground truth label of this sample
    :param loss_func: the loss function we want to maximize, i.e. the loss(.,.) function above
    :param step_size: the step size of the above gradient ascent
    :param epsilon: the attacking strength
    :param n_step: the number of steps of the above gradient ascent
    :param distance: the distance(.,.) function above
    :param bound_max: the upper bound of the adversarial sample, e.g. 0 for image data
    :param bound_min: the lower bound of the adversarial sample, e.g. 1 for image data
    :return: x_adv: the adversarial sample
    """
    device = x.device
    assert (bound_max is None and bound_min is None) or (bound_max is not None and bound_min is not None)
    if bound_max is not None and isinstance(bound_max, float):
        bound_max = torch.ones_like(x) * bound_max
    if bound_min is not None and isinstance(bound_min, float):
        bound_min = torch.ones_like(x) * bound_min

    if isinstance(model, nn.Module):
        model.eval()

    if distance == 'l_inf':
        x_adv = x.detach() + 0.001 * torch.randn_like(x).detach()    # random start
    elif distance == 'l_2':
        delta = torch.zeros_like(x).detach()
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
        x_adv = x.detach() + delta
    else:
        raise NotImplementedError(f"Unknown distance metric: {distance}")

    if bound_max is not None:
        x_adv = torch.max(torch.min(x_adv, bound_max), bound_min)  # clamp

    for _ in range(n_step):
        # perform gradient ascent
        x_adv.requires_grad_()
        with torch.enable_grad():
            output = model(x_adv)
            loss = loss_func(output, y)
        grad = torch.autograd.grad(loss, [x_adv])[0]

        # clip the adversarial noise
        if distance == 'l_inf':
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        elif distance == 'l_2':
            g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = grad / (g_norm + 1e-10)
            x_adv = x_adv.detach() + step_size * scaled_g
            delta = x_adv - x
            delta = delta.renorm(p=2, dim=0, maxnorm=epsilon)
            x_adv = x + delta
        else:
            raise NotImplementedError(f"Unknown distance metric: {distance}")

        # constrain the range of adversarial sample
        if bound_max is not None:
            x_adv = torch.max(torch.min(x_adv, bound_max), bound_min)

    return x_adv