import argparse
import logging

import torch
import torch.nn as nn
from networks import get_network
from transformers import FlattenTransformer, LinearTransformer, Polygon, LeakyReLUTransformer
from utils.loading import parse_spec
import numpy as np

DEVICE = "cpu"

torch.set_printoptions(threshold=10_000)
# logging.basicConfig(level=logging.DEBUG)

def output_size(conv, wh):
    w, h = wh
    output = [  (wh[0] + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1,
                (wh[1] + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1]
    return output


def encode_loc(tup, shape):
    residual = 0
    coefficient = 1
    for i in list(range(len(shape)))[::-1]:
        residual = residual + coefficient * tup[i]
        coefficient = coefficient * shape[i]
    return residual


def conv_linear(conv, wh):
    with torch.no_grad():
        w, h = wh
        output = output_size(conv, wh)

        in_shape = (conv.in_channels, w, h)
        out_shape = (conv.out_channels, output[0], output[1])

        fc = nn.Linear(in_features=np.prod(in_shape), out_features=np.prod(out_shape))
        fc.weight.data.fill_(0.0)

        # Output coordinates
        for x_0 in range(output[0]):
            for y_0 in range(output[1]):
                x_00 = conv.stride[0] * x_0 - conv.padding[0]
                y_00 = conv.stride[1] * y_0 - conv.padding[1]
                for xd in range(conv.kernel_size[0]):
                    for yd in range(conv.kernel_size[1]):
                        for c1 in range(conv.out_channels):
                            fc.bias[encode_loc((c1, x_0, y_0), out_shape)] = conv.bias[c1]
                            for c2 in range(conv.in_channels):
                                if 0 <= x_00 + xd < w and 0 <= y_00 + yd < h:
                                    cw = conv.weight[c1, c2, xd, yd]
                                    fc.weight[encode_loc((c1, x_0, y_0), out_shape), encode_loc((c2, x_00 + xd, y_00 + yd), in_shape)] = cw
        return fc


def analyze(
    net: torch.nn.Sequential, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    # add the 'batch' dimension
    inputs = inputs.unsqueeze(0)
    flattened = False; num_conv = 0

    input_size = [(inputs.shape[-2], inputs.shape[-1])]
    for layer in net.children():
        if not isinstance(layer, torch.nn.Conv2d):
            break
        else:
            input_size.append(tuple(output_size(layer, input_size[num_conv])))
            num_conv += 1

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
    transformer_layers = []; location = 0
    for layer in net_layers:
        if isinstance(layer, torch.nn.Flatten):
            if flattened: pass
            transformer_layers.append(FlattenTransformer())
            flattened = True
        elif isinstance(layer, torch.nn.Linear):
            transformer_layers.append(LinearTransformer(layer.weight, layer.bias))
        elif isinstance(layer, torch.nn.ReLU):
            transformer_layers.append(LeakyReLUTransformer(0.0))
        elif isinstance(layer, torch.nn.LeakyReLU):
            transformer_layers.append(LeakyReLUTransformer(layer.negative_slope))
        elif isinstance(layer, torch.nn.Conv2d):
            fc = conv_linear(layer, input_size[location])
            location += 1
            if not flattened:
                transformer_layers.append(FlattenTransformer())
                flattened = True
            transformer_layers.append(LinearTransformer(fc.weight, fc.bias))
        else:
            raise Exception(f"Unknown layer type {layer.__class__.__name__}")
    polygon_model = nn.Sequential(*transformer_layers)

    in_polygon = Polygon.create_from_input(inputs, eps=eps)
    out_polygon = polygon_model(in_polygon)
    lower_bounds, upper_bounds = out_polygon.evaluate()

    # noinspection PyTypeChecker
    verified = torch.all(lower_bounds > 0).item()
    logging.debug(f"The true label: {true_label}")
    logging.debug(f"The lower bounds: {lower_bounds}")
    logging.debug(f"The upper bounds: {upper_bounds}")
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
