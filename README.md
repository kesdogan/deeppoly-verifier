# Neural network verification with DeepPoly convex relaxation

This repository contains the code for the project of the course [Reliable and Trustworthy Artificial Intelligence](https://www.sri.inf.ethz.ch/teaching/rtai23) at ETH Zurich.

This project implements a verifier for fully-connected and convolutional neural networks using DeepPoly convex relaxation, and is implemented in Python using PyTorch.

**Authors:**

- Timur Kesdogan ([tlk13](https://github.com/tlk13))
- Juraj Mičko ([jjurm](https://github.com/jjurm))
- Evžen Wybitul ([eugleo](https://github.com/eugleo))

Read [task.md](task.md) for the original task description.

## Solution

- Standard DeepPoly convex relaxation according to:

  > G. Singh, T. Gehr, M. Püschel, and M. Vechev, ‘An abstract domain for certifying neural networks’, Proc. ACM Program. Lang., vol. 3, no. POPL, p. 41:1-41:30, Jan. 2019, doi: 10.1145/3290354.

    - In ReLU and LeakyReLU, we initialize λ's to minimize the area of the resulting shape.

- We optimize the λ's towards verifying the required property using stochastic gradient descent.


## Usage

Use the following command to run the verifier:

```bash
$ python code/verifier.py --net {net} --spec {data_directory}/{net}/img{id}_{dataset}_{eps}.txt
```

In this command,
- `net` is equal to one of the following values (each representing one of the networks we want to verify): `fc_base, fc_1, fc_2, fc_3, fc_4, fc_5, fc_6, fc_7, conv_base, conv_1, conv_2, conv_3, conv_4`.
- `data_directory` is the directory containing the test cases, i.e., `test_cases` or `preliminary_evaluation_test_cases`.
- `id` is simply a numerical identifier of the case. They are not always ordered as they have been directly sampled from a larger set of cases.
- `dataset` is the dataset name, i.e., either `mnist` or `cifar10`.
- `eps` is the perturbation that the verifier should certify in this test case.

As an example:

```bash
$ python code/verifier.py --net fc_1 --spec test_cases/fc_1/img0_mnist_0.1394.txt
```
