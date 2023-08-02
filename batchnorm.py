import torch
import torch.nn as nn
from torch.tensor import Tensor


class BatchNorm2DFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_data, gamma, beta: Tensor, running_mean,
                running_var, eps, momentum):
        # input_data : b c h w
        # eps: float or double
        # gamma :
        assert input_data.ndim == 4, f"input_data get {input_data.ndim} but expect {4}"

        mean = torch.mean(input_data, dim=(0, 2, 3), keepdim=True)
        var = torch.var(input_data, dim=(0, 2, 3), keepdim=True)

        if running_mean is not None:
            mean = mean * (1 - momentum) + running_mean.view(1, -1, 1,
                                                             1) * momentum
            running_mean[:] = mean.view(-1)

        if running_var is not None:
            var = var * (1 - momentum) + running_var.view(1, -1, 1,
                                                          1) * momentum
            running_var[:] = var.view(-1)

        x_hat = (input_data - mean) / torch.sqrt(var + eps)

        output = x_hat * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)

        ctx.eps = eps
        ctx.save_for_backward(input_data, mean, var, x_hat, gamma)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, mean, var, x_hat, gamma = ctx.saved_tensors
        eps = ctx.eps

        n = input.shape[0] * input.shape[2] * input.shape[3]

        dx_hat = gamma.view(1, -1, 1, 1) * grad_output
        dmean = torch.sum(dx_hat * (-1.0) / torch.sqrt(var + eps),
                          dim=(0, 2, 3),
                          keepdim=True)
        dvar = torch.sum(dx_hat * (input - mean) * (-0.5) *
                         torch.pow(var + eps, -1.5),
                         dim=(0, 2, 3),
                         keepdim=True)
        dx = dx_hat / torch.sqrt(var + eps) + dvar * 2.0 * (
            input - mean) / n + dmean / n

        dgamma = torch.sum(x_hat * grad_output, dim=(0, 2, 3))
        dbeta = torch.sum(1. * grad_output, dim=(0, 2, 3))

        return dx, dgamma, dbeta, None, None, None, None


def batchnorm2dfunction(input, gamma, beta, running_mean, running_var, eps,
                        momentum):
    return BatchNorm2DFunction.apply(input, gamma, beta, running_mean,
                                     running_var, eps, momentum)


class BatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, tracking=True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.tracking = tracking

    def forward(self, input):
        return batchnorm2dfunction(
            input, self.weight, self.bias,
            self.running_mean if not self.training or self.tracking else None,
            self.running_var if not self.training or self.tracking else None,
            self.eps, self.momentum)

    def extra_repr(self, input):
        return 'test'


def main():
    bn2 = nn.BatchNorm2d(3)

    bn2.train()
    bn = BatchNorm2d(3)

    bn.train()

    conv = nn.Conv2d(3, 3, 1, 1, 0)
    input = torch.rand(4, 3, 3, 3)
    print(bn2)
    #  input.cuda()

    output = bn(conv(input))
    output2 = bn2(conv(input))
    #  print(output)
    #  print(output2)
    #  print((output - output2).abs().sum())


if __name__ == "__main__":
    main()
