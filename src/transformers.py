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
            u_coefs_new = (
                torch.clamp(u_coefs, min=0) @ parent.u_coefs
                + torch.clamp(u_coefs, max=0) @ parent.l_coefs
            )

            l_bias_new = l_bias + (
                (torch.clamp(l_coefs, min=0) * parent.l_bias.unsqueeze(1)).sum(-1)
                + (torch.clamp(l_coefs, max=0) * parent.u_bias.unsqueeze(1)).sum(-1)
            )
            u_bias_new = u_bias + (
                (torch.clamp(u_coefs, min=0) * parent.u_bias.unsqueeze(1)).sum(-1)
                + (torch.clamp(u_coefs, max=0) * parent.l_bias.unsqueeze(1)).sum(-1)
            )

            l_coefs, u_coefs, l_bias, u_bias = (
                l_coefs_new,
                u_coefs_new,
                l_bias_new,
                u_bias_new,
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
            l_coefs=torch.zeros((batch, input_size, 0)),
            u_coefs=torch.zeros((batch, input_size, 0)),
            l_bias=torch.clamp(
                input_tensor.reshape((batch, input_size)) - eps, min=0, max=1
            ),
            u_bias=torch.clamp(
                input_tensor.reshape((batch, input_size)) + eps, min=0, max=1
            ),
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


class LeakyReLUTransformer(torch.nn.Module):
    negative_slope: float

    def __init__(self, negative_slope: float, init_polygon: Polygon):
        super().__init__()
        self.negative_slope = negative_slope

        # Heuristic initialization
        l_bound, u_bound = init_polygon.evaluate()
        _, n = init_polygon.l_coefs.shape[:2]
        l_coefs = torch.eye(n=n).unsqueeze(0)  # Init as y ≥ x
        l_coefs[(u_bound <= -l_bound)] *= self.negative_slope
        self.alpha = torch.nn.Parameter(l_coefs)

    def forward(self, x: Polygon) -> Polygon:
        """
        - x shape: (batch, out, in)
        - output shape: (batch, out, out)
        """
        l_bound, u_bound = x.evaluate()
        batch, n = x.l_coefs.shape[:2]

        # Initialize coefficients and biases
        l_coefs = self.alpha * torch.eye(n=n).unsqueeze(0)
        u_coefs = torch.eye(n=n).unsqueeze(0)
        l_bias = torch.zeros((batch, n))
        u_bias = torch.zeros((batch, n))

        # Always negative
        is_always_negative: torch.Tensor = u_bound <= 0
        l_coefs[is_always_negative] *= self.negative_slope
        u_coefs[is_always_negative] *= self.negative_slope

        # Always positive (values same as initialized)
        is_always_positive: torch.Tensor = l_bound >= 0

        # Crossing
        is_crossing = ~(is_always_negative | is_always_positive)
        # If negative_slope <= 1, the slope sets upper bound, otherwise sets lower bound
        slope_bound_coefs, slope_bound_bias, alpha_bound_coefs, alpha_bound_bias = (
            (u_coefs, u_bias, l_coefs, l_bias)
            if self.negative_slope <= 1
            else (l_coefs, l_bias, u_coefs, u_bias)
        )
        # Calculate the slope 𝜆 that connects the two edge points
        slope = (u_bound[is_crossing] - self.negative_slope * l_bound[is_crossing]) / (
            u_bound[is_crossing] - l_bound[is_crossing]
        )
        slope_bound_coefs[is_crossing] *= slope.unsqueeze(-1)
        slope_bound_bias[is_crossing] = l_bound[is_crossing] * (
            self.negative_slope - slope
        )
        # For alpha bound, pick the ReLU relaxation based on the minimal area heuristic
        # (the criterion is the same for LeakyReLU as for ReLU and does not depend on negative_slope)
        # Relaxation I: bound by y = negative_slope
        alpha_bound_coefs[is_crossing & (u_bound <= -l_bound)] *= self.negative_slope
        # Relaxation II: bound by y = x (values same as initialized)

        polygon = Polygon(
            l_coefs=l_coefs,
            u_coefs=u_coefs,
            l_bias=l_bias,
            u_bias=u_bias,
            parent=x,
        )
        logging.debug(f"ReLU alpha: {self.alpha}")
        logging.debug(f"ReLU output:\n{polygon}")
        return polygon
