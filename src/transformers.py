import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class Polygon:
    """
    Polygon weights have a shape (batch, out, in),
    while biases have a shape (batch, out), where
    - `out` is the dimension of the output of the layer outputting this DeepPoly, and
    - `in` is the dimension of the input of this layer
    We hereby refer to the shape of weights as the shape of the Polygon itself.
    """

    l_coefs: Tensor  # (batch, out, in)
    u_coefs: Tensor  # (batch, out, in)
    l_bias: Tensor  # (batch, out)
    u_bias: Tensor  # (batch, out)

    l_bound: Tensor  # (batch, out)
    u_bound: Tensor  # (batch, out)

    parent: Optional["Polygon"]

    def __init__(
        self,
        l_coefs: Tensor,
        u_coefs: Tensor,
        l_bias: Tensor,
        u_bias: Tensor,
        parent: Optional["Polygon"],
    ):
        self.l_coefs = l_coefs
        self.u_coefs = u_coefs
        self.l_bias = l_bias
        self.u_bias = u_bias
        self.parent = parent

        # Repeated back-substitution to calculate the bounds

        while parent is not None:
            # TODO consider early stopping if there is no dependency on the parent

            l_coefs_new = (
                torch.clamp(l_coefs, min=0) @ parent.l_coefs
                + torch.clamp(l_coefs, max=0) @ parent.u_coefs
            )
            l_coefs_new_2 = torch.zeros_like(l_coefs_new)

            u_coefs_new = (
                torch.clamp(u_coefs, min=0) @ parent.u_coefs
                + torch.clamp(u_coefs, max=0) @ parent.l_coefs
            )
            u_coefs_new_2 = torch.zeros_like(u_coefs_new)

            l_bias_new = l_bias + (
                (torch.clamp(l_coefs, min=0) * parent.l_bias.unsqueeze(1)).sum(-1)
                + (torch.clamp(l_coefs, max=0) * parent.u_bias.unsqueeze(1)).sum(-1)
            )
            l_bias_new_2 = torch.zeros_like(l_bias_new)

            u_bias_new = u_bias + (
                (torch.clamp(u_coefs, min=0) * parent.u_bias.unsqueeze(1)).sum(-1)
                + (torch.clamp(u_coefs, max=0) * parent.l_bias.unsqueeze(1)).sum(-1)
            )
            u_bias_new_2 = torch.zeros_like(u_bias_new)

            for out_i in range(l_coefs.shape[1]):
                for in_i in range(l_coefs.shape[2]):
                    coef = l_coefs[0, out_i, in_i].item()
                    if coef > 0:
                        l_coefs_new_2[0, out_i, :] += coef * parent.l_coefs[0, in_i, :]
                        l_bias_new_2[0, out_i] += coef * parent.l_bias[0, in_i]

                        u_coefs_new_2[0, out_i, :] += coef * parent.u_coefs[0, in_i, :]
                        u_bias_new_2[0, out_i] += coef * parent.u_bias[0, in_i]
                    else:
                        l_coefs_new_2[0, out_i, :] += coef * parent.u_coefs[0, in_i, :]
                        l_bias_new_2[0, out_i] += coef * parent.u_bias[0, in_i]

                        u_coefs_new_2[0, out_i, :] += coef * parent.l_coefs[0, in_i, :]
                        u_bias_new_2[0, out_i] += coef * parent.l_bias[0, in_i]

            l_coefs, u_coefs, l_bias, u_bias = (
                l_coefs_new_2,
                u_coefs_new_2,
                l_bias_new_2,
                u_bias_new_2,
            )
            parent = parent.parent

        self.l_bound = l_bias
        self.u_bound = u_bias

    def __str__(self) -> str:
        lb, ub = self.evaluate()

        result = "Polygon(\n"
        batch, out, in_ = tuple(self.l_coefs.shape)
        result += f"  shape: ({batch=}, {out=}, in={in_})\n\n"
        for b in range(batch):
            for j in range(out):
                result += f"  o{j} ∈ [{lb[b, j]}, {ub[b, j]}]\n"
                l_coefs = " + ".join(
                    f"({c} × i{i})" for i, c in enumerate(self.l_coefs[b, j])
                )
                result += f"  o{j} ≥ {l_coefs} + {self.l_bias[b, j]}\n"
                u_coefs = " + ".join(
                    f"({c} × i{i})" for i, c in enumerate(self.u_coefs[b, j])
                )
                result += f"  o{j} ≤ {u_coefs} + {self.u_bias[b, j]}\n"
                result += "\n"
        result = result.strip() + "\n)"
        return result

    @staticmethod
    def create_from_input(input_tensor: torch.Tensor, eps: float) -> "Polygon":
        """
        - input shape: (batch, source)
        - output: Polygon with shape (batch, source, 0) and bounds fixed to input_tensor +- eps
        """
        batch, *dims = input_tensor.shape
        input_size = torch.prod(torch.tensor(dims)).item()

        polygon = Polygon(
            l_coefs=torch.zeros((batch, input_size, 1)),
            u_coefs=torch.zeros((batch, input_size, 1)),
            l_bias=input_tensor.reshape((batch, input_size)) - eps,
            u_bias=input_tensor.reshape((batch, input_size)) + eps,
            # Setting parent=None will cause a trivial zero-step back substitution, which essentially sets
            # l_bound, u_bound := l_bias, u_bias
            parent=None,
        )
        logging.debug(f"Created\n{polygon}")

        return polygon

    def evaluate(self) -> tuple[Tensor, Tensor]:
        """
        - output: a tuple of lower and upper bounds, each of shape (batch, *out)
        """
        return self.l_bound, self.u_bound


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
        - x shape: (batch, out, in)
        - output shape: (batch, out_new, out)
        """
        polygon = Polygon(
            l_coefs=self.weight.unsqueeze(0),
            u_coefs=self.weight.unsqueeze(0),
            l_bias=self.bias.unsqueeze(0),
            u_bias=self.bias.unsqueeze(0),
            parent=x,
        )

        logging.debug(f"Linear layer output:\n{polygon}")
        return polygon


class ReLUTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Polygon) -> Polygon:
        """
        - x shape: (batch, out, in)
        - output shape: (batch, out, out)
        """
        l_bound, u_bound = x.evaluate()
        batch, n = x.l_coefs.shape[:2]

        l_coefs = torch.zeros((batch, n, n))
        u_coefs = torch.zeros((batch, n, n))
        l_bias = torch.zeros((batch, n))
        u_bias = torch.zeros((batch, n))

        is_always_negative: torch.Tensor = u_bound <= 0
        # In this case the output of this neuron is always 0 regardless of the network input,
        # so we clip both the lower and upper constraint inequalities to 0
        l_coefs[is_always_negative] = 0
        u_coefs[is_always_negative] = 0
        l_bias[is_always_negative] = 0
        u_bias[is_always_negative] = 0

        is_always_positive: torch.Tensor = l_bound >= 0
        # In this case the output of this neuron is always equal to the previous neuron's output
        l_coefs[is_always_positive] = torch.eye(n=n).unsqueeze(0)[is_always_positive]
        u_coefs[is_always_positive] = torch.eye(n=n).unsqueeze(0)[is_always_positive]
        l_bias[is_always_positive] = 0
        u_bias[is_always_positive] = 0

        is_crossing = ~(is_always_negative | is_always_positive)
        # DeepPoly ReLU Relaxation I
        # For lower bound we clip the inequality to 0
        l_coefs[is_crossing] = 0
        l_bias[is_crossing] = 0
        # For upper bound, we calculate the slope 𝜆 and then apply  y ≤ 𝜆 * (x − l_x)
        slope = u_bound[is_crossing] / (u_bound[is_crossing] - l_bound[is_crossing])
        u_coefs[is_crossing] = torch.eye(n=n).unsqueeze(0)[
            is_crossing
        ] * slope.unsqueeze(-1)
        u_bias[is_crossing] = -slope * l_bound[is_crossing]

        polygon = Polygon(
            l_coefs=l_coefs,
            u_coefs=u_coefs,
            l_bias=l_bias,
            u_bias=u_bias,
            parent=x,
        )
        logging.debug(f"ReLU output:\n{polygon}")
        return polygon
