import logging
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Polygon:
    """
    Polygon weights have a shape (batch, *out, *source),
    while biases have a shape (batch, *out), where
    - `*out` is the shape of the output of the layer outputting this DeepPoly, and
    - `*source` is the shape of the input, e.g. (1, 28, 28) for MNIST.
    We hereby refer to the shape of weights as the shape of the Polygon itself.
    """

    input_tensor: torch.Tensor
    eps: float

    l_coefs: Tensor  # (batch, out, source)
    l_bias: Tensor  # (batch, out)

    u_coefs: Tensor  # (batch, out, source)
    u_bias: Tensor  # (batch, out)

    def __str__(self) -> str:
        lb, ub = self.evaluate()

        result = "Polygon(\n"
        batch, out, source = tuple(self.l_coefs.shape)
        result += f"  shape: ({batch=}, {out=}, {source=})\n\n"
        for b in range(batch):
            for j in range(out):
                result += f"  o{j} ∈ [{lb[b, j]}, {ub[b, j]}]\n"
                l_coefs = " + ".join(
                    f"({c} × z{i})" for i, c in enumerate(self.l_coefs[b, j])
                )
                result += f"  o{j} ≥ {self.l_bias[b, j]} + {l_coefs}\n"
                u_coefs = " + ".join(
                    f"({c} × z{i})" for i, c in enumerate(self.u_coefs[b, j])
                )
                result += f"  o{j} ≤ {self.u_bias[b, j]} + {u_coefs}\n"
                result += "\n"
        result = result.strip() + "\n)"
        return result

    @staticmethod
    def create_from_input(input_tensor: torch.Tensor, eps: float) -> "Polygon":
        """
        - input shape: (batch, source)
        - output: Polygon with shape (batch, source, source),
        i.e. behaves like an identity layer where out == source.
        """
        batch, *dims = input_tensor.shape
        input_size = int(torch.prod(torch.tensor(dims)).item())

        polygon = Polygon(
            l_coefs=torch.eye(input_size).repeat(batch, 1, 1),
            l_bias=torch.zeros(batch, input_size),
            u_coefs=torch.eye(input_size).repeat(batch, 1, 1),
            u_bias=torch.zeros(batch, input_size),
            input_tensor=input_tensor,
            eps=eps,
        )
        logging.debug(f"Created\n{polygon}")

        return polygon

    def _get_bound(self, lower: bool) -> Tensor:
        x = self.input_tensor.reshape(self.input_tensor.shape[0], -1)

        if lower:
            weight, bias = self.l_coefs, self.l_bias
            # (batch, out, source)
            condition = weight > 0
        else:
            weight, bias = self.u_coefs, self.u_bias
            # (batch, out, source)
            condition = weight < 0

        # (batch, 1, source)
        x_lb, x_ub = (x - self.eps).unsqueeze(1), (x + self.eps).unsqueeze(1)
        # input_dim = len(x.shape) - 1

        # (batch, out, source)
        combination = torch.where(condition, x_lb, x_ub)

        # (batch, out)
        # return torch.einsum("...i,...i->...", weight, combination) + bias
        return (weight * combination).sum(-1) + bias

    def evaluate(self) -> tuple[Tensor, Tensor]:
        """
        - x.shape: (batch, *source)
        - output: a tuple of lower and upper bounds, each of shape (batch, *out)
        """
        l_bound = self._get_bound(lower=True)
        u_bound = self._get_bound(lower=False)

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
        # print((torch.clamp(self.weight, min=0) * x.l_bias.unsqueeze(1)).shape)
        # print(
        #     (
        #         (torch.clamp(self.weight, min=0) * x.l_bias.unsqueeze(1)).sum(-1)
        #         + (torch.clamp(self.weight, max=0) * x.u_bias.unsqueeze(1)).sum(-1)
        #     ).shape
        # )
        l_bias = (
            (torch.clamp(self.weight, min=0) * x.l_bias.unsqueeze(1)).sum(-1)
            + (torch.clamp(self.weight, max=0) * x.u_bias.unsqueeze(1)).sum(-1)
        ) + self.bias
        u_bias = (
            (torch.clamp(self.weight, min=0) * x.u_bias.unsqueeze(1)).sum(-1)
            + (torch.clamp(self.weight, max=0) * x.l_bias.unsqueeze(1)).sum(-1)
        ) + self.bias

        polygon = Polygon(
            l_coefs=l_coefs,
            l_bias=l_bias,
            u_coefs=u_coefs,
            u_bias=u_bias,
            input_tensor=x.input_tensor,
            eps=x.eps,
        )
        logging.debug(f"Linear layer output:\n{polygon}")
        return polygon


class ReLUTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Polygon) -> Polygon:
        """
        - x shape: (batch, in, input_size)
        - output shape: (batch, out,  input_size)
        """
        l_bound, u_bound = x.evaluate()

        l_coefs = x.l_coefs.clone()
        l_bias = x.l_bias.clone()
        u_coefs = x.u_coefs.clone()
        u_bias = x.u_bias.clone()

        is_always_negative: torch.Tensor = u_bound <= 0
        # In this case the output of this neuron is always 0 regardless of the network input,
        # so we clip both the lower and upper constraint inequalities to 0
        l_coefs[is_always_negative] = 0
        l_bias[is_always_negative] = 0
        u_coefs[is_always_negative] = 0
        u_bias[is_always_negative] = 0

        is_aways_positive: torch.Tensor = l_bound >= 0
        # In this case the output of this neuron is always equal to the previous neuron's output,
        # so we keep the lower and upper constraint inequalities unchanged

        is_crossing = ~(is_always_negative | is_aways_positive)
        # DeepPoly ReLU Relaxation I
        # For lower bound we clip the inequality to 0
        l_coefs[is_crossing] = 0
        l_bias[is_crossing] = 0
        # For upper bound, we calculate the slope 𝜆 and then apply  y ≤ 𝜆 * (x − l_x)
        slope = u_bound[is_crossing] / (u_bound[is_crossing] - l_bound[is_crossing])
        u_coefs[is_crossing] = slope.unsqueeze(-1) * x.u_coefs[is_crossing]
        u_bias[is_crossing] = slope * (x.u_bias[is_crossing] - l_bound[is_crossing])

        polygon = Polygon(
            l_coefs=l_coefs,
            l_bias=l_bias,
            u_coefs=u_coefs,
            u_bias=u_bias,
            input_tensor=x.input_tensor,
            eps=x.eps,
        )
        logging.debug(f"ReLU output:\n{polygon}")
        return polygon
