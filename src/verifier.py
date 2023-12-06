import argparse
import logging
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from networks import get_network
from transformers import (
    Conv2dTransformer,
    FlattenTransformer,
    LeakyReLUTransformer,
    LinearTransformer,
    Polygon,
)
from utils.loading import parse_spec

DEVICE = "cpu"

torch.set_printoptions(threshold=10_000)


def output_size(conv, wh):
    w, h = wh
    output = [
        (w + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1,
        (h + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1,
    ]
    return output


def analyze(
    net: torch.nn.Sequential,
    inputs: torch.Tensor,
    eps: float,
    true_label: int,
    early_stopping: bool = False,
) -> bool:
    start_time = time.time()

    # add the 'batch' dimension
    inputs = inputs.unsqueeze(0)

    n_classes = list(net.children())[-1].out_features
    final_layer = torch.nn.Linear(in_features=n_classes, out_features=n_classes - 1)

    final_layer_weights = -torch.eye(n_classes)
    final_layer_weights = torch.cat(
        (final_layer_weights[:true_label], final_layer_weights[true_label + 1 :])
    )
    final_layer_weights[:, true_label] = 1.0

    final_layer.weight.data = final_layer_weights
    final_layer.bias.data[:] = 0.0

    net_layers = list(net.children()) + [final_layer]

    # Construct a model like net that passes Polygon through each layer
    transformer_layers = []
    in_polygon: Polygon = Polygon.create_from_input(inputs, eps=eps)
    x = in_polygon
    x_size = (inputs.shape[-2], inputs.shape[-1])  # Could be a polygon property

    def add_transformer(transformer):
        nonlocal transformer_layers, x
        transformer_layers.append(transformer)
        x = transformer(x)

    for depth, layer in enumerate(net_layers):
        if isinstance(layer, torch.nn.Flatten):
            add_transformer(FlattenTransformer())
        elif isinstance(layer, torch.nn.Linear):
            add_transformer(LinearTransformer(layer.weight.data, layer.bias.data))
        elif isinstance(layer, torch.nn.ReLU):
            add_transformer(LeakyReLUTransformer(negative_slope=0.0, init_polygon=x))
        elif isinstance(layer, torch.nn.LeakyReLU):
            add_transformer(
                LeakyReLUTransformer(
                    negative_slope=layer.negative_slope, init_polygon=x
                )
            )
        elif isinstance(layer, torch.nn.Conv2d):
            # Before the first linear layer, we need to flatten the input
            if depth == 0:
                add_transformer(FlattenTransformer())
            transformer = Conv2dTransformer(
                stride=layer.stride,
                padding=layer.padding,
                kernel_size=layer.kernel_size,
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                input_size=x_size,
                weight=layer.weight.data,
                bias=layer.bias.data,  # type: ignore
            )
            add_transformer(transformer)
            x_size = transformer.output_size()
        else:
            raise Exception(f"Unknown layer type {layer.__class__.__name__}")
    polygon_model = nn.Sequential(*transformer_layers)

    verified, epochs_trained = train(
        polygon_model=polygon_model,
        in_polygon=in_polygon,
        max_epochs=100,
        early_stopping=early_stopping,
    )

    logging.info(
        f"The computation took {time.time() - start_time:.1f} seconds, {epochs_trained} epochs"
    )
    return verified


def train(
    polygon_model: torch.nn.Sequential,
    in_polygon: Polygon,
    max_epochs: int | None = None,
    early_stopping: bool = False,
) -> Tuple[bool, int]:
    trainable = len(list(polygon_model.parameters())) > 0
    optimizer = None
    if trainable:
        optimizer = torch.optim.SGD(polygon_model.parameters(), lr=2.0)

    epoch = 1
    previous_loss: Optional[torch.Tensor] = None
    while max_epochs is None or epoch <= max_epochs:
        out_polygon: Polygon = polygon_model(in_polygon)
        lower_bounds, _ = out_polygon.evaluate()

        verified: bool = torch.all(lower_bounds > 0).item()  # type: ignore
        if verified:
            return True, epoch
        if not optimizer:
            return False, epoch

        loss = lower_bounds.clamp(max=0).abs().sum()
        if early_stopping:
            if previous_loss is not None and loss >= previous_loss:
                return False, epoch
            previous_loss = loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Clamp all alpha values after each step
        for layer in polygon_model.children():
            if isinstance(layer, LeakyReLUTransformer):
                layer.clamp()

        epoch += 1

    return False, epoch


def get_gt(net, spec):
    folder = spec.split("/")[0]
    with open(f"{folder}/gt.txt", "r") as f:
        for line in f.read().splitlines():
            model, fl, answer = line.split(",")
            if model == net and fl in spec:
                return answer


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
            "fc_lecture",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    parser.add_argument(
        "--check",
        help="Whether to check the GT answer.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--early-stopping",
        help="Whether to early-stop training when loss increases.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--log",
        type=str,
    )
    args = parser.parse_args()
    if args.log:
        logging.basicConfig(level=args.log.upper())

    true_label, dataset, image, eps = parse_spec(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    verified = analyze(net, image, eps, true_label, early_stopping=args.early_stopping)
    verified_text = "verified" if verified else "not verified"
    print(verified_text)

    if args.check:
        gt = get_gt(args.net, args.spec)
        if verified_text == gt:
            print("^ correct\n")
        else:
            print(f"! incorrect, expected {gt}\n")


if __name__ == "__main__":
    main()
