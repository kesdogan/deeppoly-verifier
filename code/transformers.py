# %%
from dataclasses import dataclass

import torch
from torch import Tensor

import logging


@dataclass
class Polygon:
    """
    Polygon weights have a shape (batch, *out, *source),
    while biases have a shape (batch, *out), where
    - `*out` is the shape of the output of the layer outputting this DeepPoly, and
    - `*source` is the shape of the input, e.g. (1, 28, 28) for MNIST.
    We hereby refer to the shape of weights as the shape of the Polygon itself.
    """

    l_coefs: Tensor  # (batch, out, source)
    l_bias: Tensor  # (batch, out)

    u_coefs: Tensor  # (batch, out, source)
    u_bias: Tensor  # (batch, out)

    @staticmethod
    def create_from_input(input_shape: torch.Size) -> "Polygon":
        """
        - input shape: (batch, source)
        - output: Polygon with shape (batch, source, source),
        i.e. behaves like an identity layer where out == source.
        """
        batch, *dims = input_shape
        input_size = torch.prod(torch.tensor(dims)).item()

        polygon = Polygon(
            l_coefs=torch.eye(input_size).repeat(batch, 1),
            l_bias=torch.zeros((batch, input_size)),
            u_coefs=torch.eye(input_size).repeat(batch, 1),
            u_bias=torch.zeros((batch, input_size)),
        )
        logging.debug(f"Created Polygon {polygon}")

        return polygon

        # def _mul_last_k(self, x: Tensor, y: Tensor, k: int) -> Tensor:
        #     """
        #     Perform a scalar product on the last k dimensions of x and y.
        #     """
        #     assert x.shape == y.shape
        #     shape = x.shape[:-k]
        #     return torch.einsum("...i,...i->...", x.view(*shape, -1), y.view(*shape, -1))

    def _get_bound(self, x: Tensor, eps: float, lower: bool) -> Tensor:
        x = x.reshape(x.shape[0], -1)

        if lower:
            weight, bias = self.l_coefs, self.l_bias
            # (batch, out, source)
            condition = weight > 0
        else:
            weight, bias = self.u_coefs, self.u_bias
            # (batch, out, source)
            condition = weight < 0

        # (batch, 1, source)
        x_lb, x_ub = (x - eps).unsqueeze(1), (x + eps).unsqueeze(1)
        # input_dim = len(x.shape) - 1

        # (batch, out, source)
        combination = torch.where(condition, x_lb, x_ub)

        # (batch, out)
        # return torch.einsum("...i,...i->...", weight, combination) + bias
        return (weight * combination).sum(-1) + bias

    def evaluate(self, x: Tensor, eps: float) -> tuple[Tensor, Tensor]:
        """
        - x.shape: (batch, *source)
        - output: a tuple of lower and upper bounds, each of shape (batch, *out)
        """
        l_bound = self._get_bound(x, eps, lower=True)
        u_bound = self._get_bound(x, eps, lower=False)

        return l_bound, u_bound


class FlattenTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Polygon) -> Polygon:
        return x


class LinearTransformer(torch.nn.Module):
    weight: Tensor  # (out, in)
    bias: Tensor  # (out)

    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: Polygon) -> Polygon:
        """
        - x shape: (batch, in, input_size)
        - output shape: (batch, out,  input_size)
        """
        l_coefs = (
            torch.clamp(self.weight, min=0) @ x.l_coefs
            + torch.clamp(self.weight, max=0) @ x.u_coefs
        )
        u_coefs = (
            torch.clamp(self.weight, min=0) @ x.u_coefs
            + torch.clamp(self.weight, max=0) @ x.l_coefs
        )
        l_bias = (
            (torch.clamp(self.weight, min=0) * x.l_bias).sum(-1)
            + (torch.clamp(self.weight, max=0) * x.u_bias).sum(-1)
        ) + self.bias
        u_bias = (
            (torch.clamp(self.weight, min=0) * x.u_bias).sum(-1)
            + (torch.clamp(self.weight, max=0) * x.l_bias).sum(-1)
        ) + self.bias

        polygon = Polygon(l_coefs=l_coefs, l_bias=l_bias, u_coefs=u_coefs, u_bias=u_bias, )
        logging.debug(f"Linear layer outputs Polygon {polygon}")
        return polygon
