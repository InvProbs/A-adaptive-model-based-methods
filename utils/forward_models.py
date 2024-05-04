import torch, numbers, math
import torch.nn as nn
import torch.nn.functional as torchfunc
# from operators.operator import LinearOperator
import numpy as np


class GaussianBlur(nn.Module):
    def __init__(self, sigma, kernel_size=5, n_channels=1, n_spatial_dimensions=2):
        super(GaussianBlur, self).__init__()
        self.groups = n_channels
        if isinstance(kernel_size, numbers.Number):
            self.padding = int(math.floor(kernel_size / 2))
            kernel_size = [kernel_size] * n_spatial_dimensions
        else:
            print('KERNEL SIZE MUST BE A SINGLE INTEGER - RECTANGULAR KERNELS NOT SUPPORTED AT THIS TIME')
            exit()
        self.gaussian_kernel = torch.nn.Parameter(self.create_gaussian_kernel(sigma, kernel_size, n_channels),
                                                  requires_grad=False)

    def create_gaussian_kernel(self, sigma, kernel_size, n_channels):
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, mgrid in zip(kernel_size, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) / sigma) ** 2 / 2)

        # Make sure norm of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(n_channels, *[1] * (kernel.dim() - 1))
        return kernel

    def forward(self, x):
        return torchfunc.conv2d(x, weight=self.gaussian_kernel, groups=self.groups, padding=self.padding)

class noisyGaussianBlur(nn.Module):
    def __init__(self, sigma, kernel_size=5, n_channels=1, n_spatial_dimensions=2):
        super(noisyGaussianBlur, self).__init__()
        self.groups = n_channels
        if isinstance(kernel_size, numbers.Number):
            self.padding = int(math.floor(kernel_size / 2))
            kernel_size = [kernel_size] * n_spatial_dimensions
        else:
            print('KERNEL SIZE MUST BE A SINGLE INTEGER - RECTANGULAR KERNELS NOT SUPPORTED AT THIS TIME')
            exit()
        self.gaussian_kernel = torch.nn.Parameter(self.create_gaussian_kernel(sigma, kernel_size, n_channels),
                                                  requires_grad=False)

    def create_gaussian_kernel(self, sigma, kernel_size, n_channels):
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, mgrid in zip(kernel_size, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) / sigma) ** 2 / 2)

        # Make sure norm of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(n_channels, *[1] * (kernel.dim() - 1))
        return kernel + torch.randn_like(kernel) * 0.001

    def forward(self, x):
        return torchfunc.conv2d(x, weight=self.gaussian_kernel, groups=self.groups, padding=self.padding)