import torch
import torch.nn as nn
import utils.spectral_norm as chen


class netA(nn.Module):
    def __init__(self, args, nlayers):
        super(netA, self).__init__()
        layers = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False))

        self.data_layer = nn.Sequential(*layers)
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

    def forward(self, xk, A0):
        y0 = A0 @ xk
        yk = y0 + self.data_layer(y0)
        return y0, yk


class netA_kernel(nn.Module):
    def __init__(self, args, nlayers):
        super(netA_kernel, self).__init__()
        layers = [nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1, bias=False),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False))

        self.data_layer = nn.Sequential(*layers)
        self.feature_layer = nn.Sequential(
            nn.Linear(51, 1600),  # 32x50
            nn.BatchNorm1d(1600),
            nn.GELU(),
            nn.Linear(1600, 6400),  # 128x50
            nn.BatchNorm1d(6400),
        )
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

    def forward(self, xk, A0):
        bs, C, H, W = xk.shape
        w = A0[:, 0, :51, 20]
        features = self.feature_layer(w).reshape(bs, 1, H, W)  # [128, 1, 128, 50]
        y0 = A0 @ xk  # [128, 1, 128, 50]
        input_stack = torch.cat((y0, features), dim=1)
        yk = y0 + self.data_layer(input_stack)
        return y0, yk


class netA_foggy(nn.Module):
    def __init__(self, args):
        super(netA_foggy, self).__init__()
        ks = args.kernel_size
        pad = args.padding
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=ks, padding=pad),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=ks, padding=pad),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=ks, padding=pad),
        )
        self.light_layer = nn.Sequential(*model)
        self.trans_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=ks, padding=pad),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv2d(64, 3, kernel_size=ks, padding=pad),
            nn.Sigmoid(),
        )
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

    def forward(self, xk, D):
        """
        Simulation of adding fog
        y = x * T(x) + A (1-T(x))
        T(x): between 0 and 1, transmission map
        A: atmospheric light
        """
        T = self.trans_layer(D)
        yk = xk * T + self.light_layer(xk) * (torch.ones_like(T).to('cuda') - T)
        y0 = xk
        return y0, yk


class netA_deblur(nn.Module):
    def __init__(self, args, nlayers):
        super(netA_deblur, self).__init__()
        layers = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False))

        self.data_layer = nn.Sequential(*layers)
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

    def forward(self, xk, A0):
        # bs, C, H, W = xk.shape
        y0 = A0(xk)
        yk = y0 + self.data_layer(y0)
        return y0, yk


class proxNet_deconv(nn.Module):
    def __init__(self, args, nlayers):
        super(proxNet_deconv, self).__init__()
        layers = [nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1, bias=False),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False))

        self.R = nn.Sequential(*layers)
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

    def forward(self, zk, xk, y):
        return self.R(torch.cat((zk - torch.exp(self.eta) * (zk - xk), y), dim=1))


class proxNet_fog(nn.Module):
    def __init__(self, nlayers, args):
        super(proxNet_fog, self).__init__()
        layers = [nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, padding=1, bias=False),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False))

        self.R = nn.Sequential(*layers)
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

    def forward(self, zk, xk, y):
        return self.R(torch.cat((zk - torch.exp(self.eta) * (zk - xk), y), dim=1))


class proxNet_deblur(nn.Module):
    def __init__(self, args, nlayers):
        super(proxNet_deblur, self).__init__()
        layers = [nn.Conv2d(in_channels=3 * 2, out_channels=64, kernel_size=3, padding=1, bias=False),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False))

        self.R = nn.Sequential(*layers)
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

    def forward(self, Xk, Zk, y):
        return self.R(torch.cat((Xk - torch.exp(self.eta) * (Xk - Zk), y), dim=1))


class netA_kernel_sn(nn.Module):
    def __init__(self, args, nlayers):
        super(netA_kernel_sn, self).__init__()
        layers = [chen.spectral_norm(nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1, bias=False)),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(
                chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(
            chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False)))

        self.data_layer = nn.Sequential(*layers)
        self.feature_layer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(51, 1600)),  # 32x50
            nn.BatchNorm1d(1600),
            nn.GELU(),
            nn.utils.spectral_norm(nn.Linear(1600, 6400)),  # 128x50
            nn.BatchNorm1d(6400),
        )
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

    def forward(self, xk, A0):
        bs, C, H, W = xk.shape
        w = A0[:, 0, :51, 20]
        features = self.feature_layer(w).reshape(bs, 1, H, W)  # [128, 1, 128, 50]
        y0 = A0 @ xk  # [128, 1, 128, 50]
        input_stack = torch.cat((y0, features), dim=1)
        yk = y0 + self.data_layer(input_stack)
        return y0, yk


class proxNet_deconv_sn(nn.Module):
    def __init__(self, args, nlayers):
        super(proxNet_deconv_sn, self).__init__()
        layers = [chen.spectral_norm(nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1, bias=False)),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(
                chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(
            chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False)))

        self.R = nn.Sequential(*layers)
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

    def forward(self, zk, xk, y):
        return self.R(torch.cat((zk - torch.exp(self.eta) * (zk - xk), y), dim=1))


class netA_deblur_sn(nn.Module):
    def __init__(self, args, nlayers):
        super(netA_deblur_sn, self).__init__()
        layers = [chen.spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(
                chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(
            chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, bias=False)))

        self.data_layer = nn.Sequential(*layers)
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

    def forward(self, xk, A0):
        # bs, C, H, W = xk.shape
        y0 = A0(xk)
        yk = y0 + self.data_layer(y0)
        return y0, yk


class proxNet_deblur_sn(nn.Module):
    def __init__(self, args, nlayers):
        super(proxNet_deblur_sn, self).__init__()
        layers = [
            chen.spectral_norm(nn.Conv2d(in_channels=3 * 2, out_channels=64, kernel_size=3, padding=1, bias=False)),
            nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(
                chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(
            chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False)))

        self.R = nn.Sequential(*layers)
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

    def forward(self, Xk, Zk, y):
        return self.R(torch.cat((Xk - torch.exp(self.eta) * (Xk - Zk), y), dim=1))


class netA_fogging_sn(nn.Module):
    def __init__(self, args):
        super(netA_fogging_sn, self).__init__()
        ks = args.kernel_size
        pad = args.padding
        model = nn.Sequential(
            chen.spectral_norm(nn.Conv2d(3, 64, kernel_size=ks, padding=pad)),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            chen.spectral_norm(nn.Conv2d(64, 64, kernel_size=ks, padding=pad)),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            chen.spectral_norm(nn.Conv2d(64, 3, kernel_size=ks, padding=pad)),
        )
        self.light_layer = nn.Sequential(*model)
        self.trans_layer = nn.Sequential(
            chen.spectral_norm(nn.Conv2d(1, 64, kernel_size=ks, padding=pad)),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            chen.spectral_norm(nn.Conv2d(64, 3, kernel_size=ks, padding=pad)),
            nn.Sigmoid(),
        )
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

    def forward(self, xk, D):
        """
        Simulation of adding fog
        y = x * T(x) + A (1-T(x))
        T(x): between 0 and 1, transmission map
        A: atmospheric light
        """
        T = self.trans_layer(D)
        y0 = xk
        yk = xk * T + self.light_layer(xk) * (torch.ones_like(T).to('cuda') - T)
        return y0, yk


class proxNet_fog_sn(nn.Module):
    def __init__(self, nlayers, args):
        super(proxNet_fog_sn, self).__init__()
        layers = [chen.spectral_norm(nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, padding=1, bias=False)),
                  nn.GELU()]
        for _ in range(nlayers - 2):
            layers.append(
                chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False)))
            layers.append(nn.GroupNorm(4, 64))
            layers.append(nn.GELU())
        layers.append(
            chen.spectral_norm(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, bias=False)))

        self.R = nn.Sequential(*layers)
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

    def forward(self, zk, xk, y):
        return self.R(torch.cat((zk - torch.exp(self.eta) * (zk - xk), y), dim=1))
