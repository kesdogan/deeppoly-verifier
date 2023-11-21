import argparse
import logging

import torch
import torch.nn as nn

from transformers import LinearTransformer, FlattenTransformer, Polygon
from networks import get_network
from utils.loading import parse_spec

DEVICE = "cpu"

torch.set_printoptions(threshold=10_000)
logging.basicConfig(level=logging.DEBUG)


def analyze(
    net: torch.nn.Sequential, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    # add the 'batch' dimension
    inputs = inputs.unsqueeze(0)

    # Construct a model like net that passes Polygon through each layer
    layers = []
    for layer in net.children():
        if isinstance(layer, torch.nn.Flatten):
            layers.append(FlattenTransformer())
        if isinstance(layer, torch.nn.Linear):
            layers.append(LinearTransformer(layer.weight, layer.bias))
    polygon_model = nn.Sequential(*layers)

    in_polygon = Polygon.create_from_input(inputs.shape)
    out_polygon = polygon_model(in_polygon)
    lower_bounds, upper_bounds = out_polygon.evaluate(inputs, eps)

    # noinspection PyTypeChecker
    verified = torch.all(
        torch.cat(
            (upper_bounds[:, :true_label], upper_bounds[:, (true_label + 1) :]), dim=1
        )
        < lower_bounds[:, true_label]
    ).item()

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
