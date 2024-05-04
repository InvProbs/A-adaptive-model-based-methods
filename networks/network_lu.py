import torch.nn as nn
import torch


class LU_deconv(nn.Module):
    def __init__(self, args, nlayers):
        super(LU_deconv, self).__init__()
        layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False))
        # self.dncnn = nn.Sequential(*layers)
        self.prox = nn.Sequential(*layers)
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            with torch.no_grad():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.001)
                    m.weight /= 10
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, xk, y, A):
        grad = - torch.transpose(A, 3, 2) @ (y - A @ xk)
        return self.prox(xk - torch.exp(self.eta) * grad)


class LU_deconv_defog(nn.Module):
    def __init__(self, args, nlayers):
        super(LU_deconv_defog, self).__init__()
        layers = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False))
        # self.dncnn = nn.Sequential(*layers)
        self.prox = nn.Sequential(*layers)
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            with torch.no_grad():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.001)
                    m.weight /= 10
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, xk, y):
        grad = - (y - xk)
        return self.prox(xk - torch.exp(self.eta) * grad)
    
    
class LU_deblur(nn.Module):
    def __init__(self, args, nlayers):
        super(LU_deblur, self).__init__()
        layers = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False))
        self.prox = nn.Sequential(*layers)
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            with torch.no_grad():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.001)
                    m.weight /= 10
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, xk, y, Ak):
        ls_grad = xk - torch.exp(self.eta) * Ak(Ak(xk)) - Ak(y)  # the adjoint of adding blur is same as adding blur
        return self.prox(ls_grad)
