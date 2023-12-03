import argparse
import multiprocessing
from pprint import pprint

import torch
import torch.nn as nn
from networks import get_network
from transformers import (
    FlattenTransformer,
    LeakyReLUTransformer,
    LinearTransformer,
    Polygon,
)
from utils.loading import parse_spec

DEVICE = "cpu"

torch.set_printoptions(threshold=10_000)
# # logging.basicConfig(level=logging.WARNING)


def analyze(
    net: torch.nn.Sequential, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
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
    for layer in net_layers:
        if isinstance(layer, torch.nn.Flatten):
            transformer = FlattenTransformer()
            transformer_layers.append(transformer)
            x = transformer(x)
        elif isinstance(layer, torch.nn.Linear):
            weight, bias = layer.weight.data, layer.bias.data
            transformer = LinearTransformer(weight, bias)
        elif isinstance(layer, torch.nn.ReLU):
            transformer = LeakyReLUTransformer(negative_slope=0.0, init_polygon=x)
        elif isinstance(layer, torch.nn.LeakyReLU):
            transformer = LeakyReLUTransformer(
                negative_slope=layer.negative_slope, init_polygon=x
            )
        else:
            raise Exception(f"Unknown layer type {layer.__class__.__name__}")
        x = transformer(x)
        transformer_layers.append(transformer)
    polygon_model = nn.Sequential(*transformer_layers)

    # TODO Run 10 agents
    verified = train(polygon_model=polygon_model, in_polygon=in_polygon, epochs=1000)

    return verified


def train(
    polygon_model: torch.nn.Sequential,
    in_polygon: Polygon,
    epochs: int | None = None,
):
    optimizer = torch.optim.Adam(polygon_model.parameters())

    epoch = 1
    verified = False
    # TODO Maybe check the delta in loss to stop early?
    # TODO ...but only for testing, bc in submission we can safely time out
    while not verified and (epochs is None or epoch <= epochs):
        out_polygon: Polygon = polygon_model(in_polygon)
        lower_bounds, _ = out_polygon.evaluate()
        loss = lower_bounds.clamp(max=0).abs().sum()
        print(f"Epoch: {epoch:4d}, loss: {loss.item():.3f}")

        verified: bool = torch.all(lower_bounds > 0).item()  # type: ignore

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch += 1

    return verified


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
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    verified = analyze(net, image, eps, true_label)
    verified_text = "verified" if verified else "not verified"
    print(verified_text)

    if args.check:
        with open("test_cases/gt.txt", "r") as f:
            for line in f.read().splitlines():
                model, fl, answer = line.split(",")
                if model == args.net and fl in args.spec:
                    if verified_text == answer:
                        print("^ correct\n")
                    else:
                        print(f"incorrect, expected {answer}\n")


if __name__ == "__main__":
    main()
