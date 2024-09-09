from __future__ import print_function
import argparse
import torch
import torch.ao.quantization
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# This model comes from https://colab.research.google.com/drive/1oDfcLRz2AIgsclkXJHj-5wMvbylr4Nxz#scrollTo=xhiL7OwwuLS6
class Net(nn.Module):
    def __init__(self, mnist=True):

        super(Net, self).__init__()
        if mnist:
            num_channels = 1
        else:
            num_channels = 3

        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.contiguous().view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader):
    if isinstance(model, nn.Module):
        model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


dataset_path = "mnist-dataset"


def main():
    batch_size = 64
    test_batch_size = 64
    epochs = 10
    lr = 0.01
    momentum = 0.5
    seed = 1
    log_interval = 500
    no_cuda = False

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        default=False,
        help="Use pretrained model weights",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default="mnist_cnn.pt",
        metavar="PATH",
        help="Path to pretrained model weights (default: mnist_cnn.pt)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="mnist-dataset",
        metavar="PATH",
        help="Path to the dataset (default: mnist-dataset)",
    )
    args = parser.parse_args()

    save_model = args.save_model
    use_pretrained = args.use_pretrained
    pretrained_weights = args.pretrained_weights
    global dataset_path
    dataset_path = args.dataset_path

    if use_pretrained:
        model = Net()
        model.load_state_dict(torch.load(pretrained_weights, weights_only=True))
        model.eval()
        return model

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    # See https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            dataset_path,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    args = {}
    args["log_interval"] = log_interval
    for epoch in range(1, epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return model


import copy
from torch.quantization import quantize_fx


# Dump from model_quantized, only for debug usage
def forward(self, x):
    conv1_input_scale_0 = self.conv1_input_scale_0
    conv1_input_zero_point_0 = self.conv1_input_zero_point_0
    quantize_per_tensor = torch.quantize_per_tensor(
        x, conv1_input_scale_0, conv1_input_zero_point_0, torch.quint8
    )
    conv1 = self.conv1(quantize_per_tensor)
    max_pool2d = torch.nn.functional.max_pool2d(
        conv1, 2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=False
    )
    conv2 = self.conv2(max_pool2d)
    max_pool2d = None
    max_pool2d_1 = torch.nn.functional.max_pool2d(
        conv2, 2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=False
    )
    contiguous = max_pool2d_1.contiguous()
    view = contiguous.view(-1, 800)
    fc1 = self.fc1(view)
    fc2 = self.fc2(fc1)
    dequantize_8 = fc2.dequantize()
    log_softmax = torch.nn.functional.log_softmax(dequantize_8, dim=1, _stacklevel=3, dtype=None)
    return log_softmax


if __name__ == "__main__":
    model = main()
    backend = "fbgemm"
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            dataset_path,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )
    model(torch.randn(1, 1, 28, 28))
    m = copy.deepcopy(model)
    m.eval()
    qconfig_mapping = torch.ao.quantization.get_default_qat_qconfig_mapping(backend)
    # Prepare
    model_prepared = quantize_fx.prepare_fx(m, qconfig_mapping, torch.randn(1, 1, 28, 28))
    # Calibrate - Use representative (validation) data.
    with torch.inference_mode():
        for i, (data, target) in enumerate(test_loader):
            # Use only 100 samples for calibration
            if i > 100:
                testd, testt = data, target
                break
            model_prepared(data)

    # Quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)
    print("Original Model:")
    test({}, model, "cpu", test_loader)
    print("Quantized Model:")
    test({}, model_quantized, "cpu", test_loader)
    # TODO: get quantized weights
    from quantized_layer import QuantizedConvReLU2d, QuantizedLinearReLU
    from torch.ao.nn.intrinsic.quantized.modules.conv_relu import ConvReLU2d
    from torch.ao.nn.intrinsic.quantized.modules.linear_relu import LinearReLU
    from torch.ao.nn.quantized.modules.linear import Linear

    input_scale = model_quantized.conv1_input_scale_0.item()
    input_zero_point = model_quantized.conv1_input_zero_point_0.item()
    net = []
    for name, layer in model_quantized.named_children():
        if isinstance(layer, ConvReLU2d):
            # TODO: Consider groups, padding, dilation, etc.
            q = QuantizedConvReLU2d(
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                qw=layer.weight(),
                bias=layer.bias(),
                input_scale=input_scale,
                input_zero=input_zero_point,
                output_scale=layer.scale,
                output_zero=layer.zero_point,
            )
            net.append(q)
            # TODO: How to extract other layers?
            net.append("pool")
            input_scale = layer.scale
            input_zero_point = layer.zero_point
        elif isinstance(layer, (Linear, LinearReLU)):
            q = QuantizedLinearReLU(
                in_features=layer.in_features,
                out_features=layer.out_features,
                input_scale=input_scale,
                input_zero=input_zero_point,
                output_scale=layer.scale,
                output_zero=layer.zero_point,
                qw=layer.weight(),
                bias=layer.bias(),
            )
            input_scale = layer.scale
            input_zero_point = layer.zero_point
            net.append(q)

    def run(t):
        for layer in net:
            if layer == "pool":
                t = torch.nn.functional.max_pool2d(
                    t, 2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=False
                )
            else:
                t = layer(t)
        return torch.nn.functional.log_softmax(t.dequantize(), dim=1, _stacklevel=3, dtype=None)

    print("Manually quantized Model:")
    # TODO: Figure out why there is a difference between the two quantized models
    # Maybe I need to see the implementation of torch.ops.quantized.* functions?
    # See https://discuss.pytorch.org/t/how-to-implement-forward-pass-for-a-quantized-linear/150699
    test({}, run, "cpu", test_loader)
