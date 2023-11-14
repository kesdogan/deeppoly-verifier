# %%
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

    l_coefs: Tensor  # (batch, *out, *source)
    l_bias: Tensor  # (batch, *out)

    u_coefs: Tensor  # (batch, *out, *source)
    u_bias: Tensor  # (batch, *out)

    @staticmethod
    def create_from_input(x: Tensor) -> "Polygon":
        """
        - input shape: (batch, *source)
        - output: Polygon with shape (batch, *source, *source),
        i.e. behaves like an identity layer where *out == *source.
        """
        batch, *input_shape = x.shape
        filler_shape = [1 for _ in input_shape]
        N = torch.prod(torch.tensor(input_shape)).item()
        p_tensor = torch.arange(N).reshape(*input_shape, *filler_shape)
        i_tensor = torch.arange(N).reshape(*filler_shape, *input_shape)

        eye = (p_tensor == i_tensor).float()

        return Polygon(
            l_coefs=eye.repeat(batch, 1),
            l_bias=torch.zeros_like(x),
            u_coefs=eye.repeat(batch, 1),
            u_bias=torch.zeros_like(x),
        )

    def _mul_last_k(self, x: Tensor, y: Tensor, k: int) -> Tensor:
        """
        Perform a scalar product on the last k dimensions of x and y.
        """
        assert x.shape == y.shape
        shape = x.shape[:-k]
        return torch.einsum("...i,...i->...", x.view(*shape, -1), y.view(*shape, -1))

    def _get_bound(self, x: Tensor, eps: float, lower: bool) -> Tensor:
        if lower:
            weight, bias = self.l_coefs, self.l_bias
            condition = weight > 0
        else:
            weight, bias = self.u_coefs, self.u_bias
            condition = weight < 0

        x_lb, x_ub = (x - eps).unsqueeze(1), (x + eps).unsqueeze(1)
        input_dim = len(x.shape) - 1

        # (batch, *out, *source)
        combination = torch.where(condition, x_lb, x_ub)

        # (batch, *out)
        return self._mul_last_k(weight, combination, k=input_dim) + bias

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
        batch_size = x.l_coefs.shape[0]
        return Polygon(
            l_coefs=x.l_coefs.reshape(batch_size, -1),
            l_bias=x.l_bias,
            u_coefs=x.u_coefs,
            u_bias=x.u_bias,
        )


class LinearTransformer(torch.nn.Module):
    # neuron = SUM_i ( weight_i * input_i ) + bias

    # parameters from the network:
    # - weight[m, n]
    # - bias[n]

    # i - iterates over input space
    # n - iterates over the current layer
    # m - iterates over the previous layer

    # l_constraint[m] = SUM_i ( l_coefs[m, i] * input_i ) + l_bias[m]
    # u_constraint[m] = SUM_i ( u_coefs[m, i] * input_i ) + u_bias[m]

    # neuron >= SUM_n ( weight_i * (l_constraints or u_constraints) ) + bias
    # neuron <= SUM_n ( weight_i * (u_constraints or l_constraints) ) + bias

    # l_constraint[n] = SUM_m (
    #               weight[m, n]
    #               * ( [weight[m, n] > 0] * l_constraint[m] + [weight[m, n] <= 0] * u_constraint[m] )
    #           ) + bias[n]
    # u_constraint[n] = SUM_m (
    #               weight[m, n]
    #               * ( [weight[m, n] > 0] * u_constraint[m] + [weight[m, n] <= 0] * l_constraint[m] )
    #           ) + bias[n]

    # l_coefs[n, i] = SUM_m (
    #     weight[m, n]
    #     * ( [weight[m, n] > 0] * l_coefs[m, i] + [weight[m, n] <= 0] * u_coefs[m, i] )
    #     )

    # l_bias[n] = SUM_m (
    #     weight[m, n]
    #     * ( [weight[m, n] > 0] * l_bias[m] + [weight[m, n] <= 0] * u_bias[m] )
    #     ) + bias[n]

    # l_coefs_2[n, i] = SUM_m (
    #     weight[m, n]
    #     * l_coefs[m, i]
    #     )

    # for a neuron: l_coefs[i] = weight * SUM_m ( [weight[m, i] > 0] * l_coefs[m, i] + [weight[m, i] <= 0] * u_coefs[m, i] )

    weight: Tensor  # (in_size, out_size)
    bias: Tensor  # (out_size)

    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x: Polygon) -> Polygon:
        """
        - x shape: (batch, in_size, *source)
        - output shape: (batch, out_size, *source)
        """

        # TODO Shapes don't make sense
        # (out_size, in_size, *source)
        l_coefs = torch.where(
            self.weight.unsqueeze(-1) > 0,  # (out_size, in_size, 1)
            x.l_coefs,  # (in_size, *source)
            x.u_coefs,  # (in_size, *source)
        )
        # (out_size, *source)
        l_coefs_1 = self.weight.T @ l_coefs

        l_coefs_2 = (
            torch.clamp(self.weight.T, 0, None) @ x.l_coefs
            + torch.clamp(self.weight.T, None, 0) @ x.u_coefs
        )
        assert l_coefs_1 == l_coefs_2
        l_coefs = l_coefs_1

        u_coefs = self.weight.T @ torch.where(
            self.weight.T.unsqueeze(-1) > 0, x.u_coefs, x.l_coefs
        )

        l_bias = self.weight.T @ x.l_coefs + self.bias
        u_bias = self.weight.T @ x.u_coefs + self.bias

        l_bound = l_coefs @ self.inputs
