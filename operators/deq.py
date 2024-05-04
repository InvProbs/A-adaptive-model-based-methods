import torch
import torch.nn as nn
import torch.autograd as autograd


class deblur_deq_kth_iter(nn.Module):
    def __init__(self, args, netR, netA, optR, optA, criteria):
        super(deblur_deq_kth_iter, self).__init__()
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self.R = netR
        self.netA = netA
        self.optR = optR
        self.optA = optA
        self.lr_Z = args.lr_Z
        self.criteria = criteria
        self.gamma = args.gamma
        self.lamb = args.lamb
        self.Zk = []
        self.A0 = []

    def init_setup(self, A0, Zk, X0):
        self.A0, self.Zk = A0, Zk
        self.X0 = X0

    def set_Zk(self, Zk):
        self.Zk = Zk

    def _linear_op(self, x):
        return self.A0(x)

    def _linear_adjoint(self, x):
        return self.A0(x)

    def get_gradient(self, z, y):
        return self._linear_adjoint(self._linear_op(z)) - self._linear_adjoint(y) - self.R(z)

    def forward(self, Xk, y, no_grad=True):
        # update X
        self.netA.eval()
        self.R.eval()
        optZ = torch.optim.AdamW([self.Zk], lr=self.lr_Z)
        self.Zk.requires_grad_(True)
        yinit, yk = self.netA(self.Zk, self.A0)
        optZ.zero_grad()
        lossZ = self.criteria(y, yk) + self.gamma * self.criteria(yk, yinit) + self.lamb * self.criteria(Xk.detach(),
                                                                                                         self.Zk)
        lossZ.backward()
        optZ.step()
        Zk = self.Zk.detach()

        # update theta in netA_theta
        self.netA.train()
        self.R.eval()
        Zk.requires_grad_(False)
        yinit, yk = self.netA(Zk, self.A0)
        self.optA.zero_grad()
        lossA = self.criteria(y, yk) + self.gamma * self.criteria(yk, yinit)
        lossA.backward()
        self.optA.step()

        # update X
        if no_grad:
            with torch.no_grad():
                self.netA.eval()
                self.R.train()
                Xk = self.R(Xk - self.X0, Zk, y)
        else:
            self.netA.eval()
            self.R.train()
            Xk = self.R(Xk - self.X0, Zk, y)
        return Xk + self.X0


#
# class deconv_deq_kth_iter(nn.Module):
#     def __init__(self, args, netR, netA, optR, optA, criteria):
#         super(deconv_deq_kth_iter, self).__init__()
#         self.eta = nn.Parameter(torch.ones(1) * args.eta)
#         self.R = netR
#         self.netA = netA
#         self.optR = optR
#         self.optA = optA
#         self.lr_Z = args.lr_Z
#         self.criteria = criteria
#         self.gamma = args.gamma
#         self.lamb = args.lamb
#         self.Zk = []
#         self.A0 = []
#
#     def init_setup(self, A0, Zk, X0):
#         self.A0, self.Zk = A0, Zk
#         self.X0 = X0
#
#     def set_Zk(self, Zk):
#         self.Zk = Zk
#
#     def forward(self, Xk, y, no_grad=True):
#         # update X
#         self.netA.eval()
#         self.R.eval()
#         optZ = torch.optim.AdamW([self.Zk], lr=self.lr_Z)
#         self.Zk.requires_grad_(True)
#         yinit, yk = self.netA(self.Zk, self.A0)
#         optZ.zero_grad()
#         lossZ = self.criteria(y, yk) + self.gamma * self.criteria(yk, yinit) + self.lamb * self.criteria(Xk.detach(), self.Zk)
#         lossZ.backward()
#         optZ.step()
#         Zk = self.Zk.detach()
#
#         # update theta in netA_theta
#         self.netA.train()
#         self.R.eval()
#         Zk.requires_grad_(False)
#         yinit, yk = self.netA(Zk, self.A0)
#         self.optA.zero_grad()
#         lossA = self.criteria(y, yk) + self.gamma * self.criteria(yk, yinit)
#         lossA.backward()
#         self.optA.step()
#
#         # update X
#         if no_grad:
#             with torch.no_grad():
#                 self.netA.eval()
#                 self.R.train()
#                 Xk = self.R(Xk, Zk, y)
#         else:
#             self.netA.eval()
#             self.R.train()
#             Xk = self.R(Xk, Zk, y)
#         return Xk

class deconv_deq_kth_iter(nn.Module):
    def __init__(self, args, netR, netA, optR, optA, criteria):
        super(deconv_deq_kth_iter, self).__init__()
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self.R = netR
        self.netA = netA
        self.optR = optR
        self.optA = optA
        self.lr_Z = args.lr_Z
        self.criteria = criteria
        self.gamma = args.gamma
        self.lamb = args.lamb
        self.Zk = []
        self.A0 = []

    def init_setup(self, A0, Zk, X0):
        self.A0, self.Zk = A0, Zk
        self.X0 = X0

    def set_Zk(self, Zk):
        self.Zk = Zk

    def forward(self, Xk, y, no_grad=True):
        # update X
        self.netA.eval()
        self.R.eval()
        optZ = torch.optim.AdamW([self.Zk], lr=self.lr_Z)
        self.Zk.requires_grad_(True)
        yinit, yk = self.netA(self.Zk, self.A0)
        optZ.zero_grad()
        lossZ = self.criteria(y, yk) + self.gamma * self.criteria(yk, yinit) + self.lamb * self.criteria(Xk.detach(),
                                                                                                         self.Zk)
        lossZ.backward()
        optZ.step()
        Zk = self.Zk.detach()

        # update theta in netA_theta
        self.netA.train()
        self.R.eval()
        Zk.requires_grad_(False)
        yinit, yk = self.netA(Zk, self.A0)
        self.optA.zero_grad()
        lossA = self.criteria(y, yk) + self.gamma * self.criteria(yk, yinit)
        lossA.backward()
        self.optA.step()

        # update X
        if no_grad:
            with torch.no_grad():
                self.netA.eval()
                self.R.train()
                Xk = self.R(Xk - self.X0, Zk, y)
                # Xk = self.R(Xk, Zk, y)
        else:
            self.netA.eval()
            self.R.train()
            Xk = self.R(Xk - self.X0, Zk, y)
            # Xk = self.R(Xk, Zk, y)
        return Xk #+ self.X0


class defog_deq_kth_iter(nn.Module):
    def __init__(self, args, netR, netA, optR, optA, criteria):
        super(defog_deq_kth_iter, self).__init__()
        self.eta = nn.Parameter(torch.ones(1) * args.eta)
        self.R = netR
        self.netA = netA
        self.optR = optR
        self.optA = optA
        self.lr_Z = args.lr_Z
        self.criteria = criteria
        self.gamma = args.gamma
        self.lamb = args.lamb
        self.Zk = []
        self.A0 = []

    def init_setup(self, A0, Zk, X0):
        self.A0, self.Zk = A0, Zk
        self.X0 = X0

    def set_Zk(self, Zk):
        self.Zk = Zk

    def forward(self, Xk, y, no_grad=True):
        # update X
        self.netA.eval()
        self.R.eval()
        optZ = torch.optim.AdamW([self.Zk], lr=self.lr_Z)
        self.Zk.requires_grad_(True)
        yinit, yk = self.netA(self.Zk, self.A0)
        optZ.zero_grad()
        lossZ = self.criteria(y, yk) + self.gamma * self.criteria(yk, yinit) + self.lamb * self.criteria(Xk.detach(),
                                                                                                         self.Zk)
        lossZ.backward()
        optZ.step()
        Zk = self.Zk.detach()

        # update theta in netA_theta
        self.netA.train()
        self.R.eval()
        Zk.requires_grad_(False)
        yinit, yk = self.netA(Zk, self.A0)
        self.optA.zero_grad()
        lossA = self.criteria(y, yk) + self.gamma * self.criteria(yk, yinit)
        lossA.backward()
        self.optA.step()

        # update X
        if no_grad:
            with torch.no_grad():
                self.netA.eval()
                self.R.train()
                # Xk = self.R(Xk, Zk, y)
                Xk = self.R(Xk - self.X0, Zk, y)
        else:
            self.netA.eval()
            self.R.train()
            # Xk = self.R(Xk, Zk, y)
            Xk = self.R(Xk - self.X0, Zk, y)
        return Xk #+ self.X0


""" define DEQ specs """


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration.
    This was taken from the Deep Equilibrium tutorial here: http://implicit-layers-tutorial.org/deep_equilibrium_models/
    """

    # global tt
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape(x0.shape)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    current_k = 0
    past_iterate = x0
    for k in range(2, max_iter):
        current_k = k
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        # alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n) NEW

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        current_iterate = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape(x0.shape)).reshape(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        # res.append((X[:, k % m]).view_as(x0))

        if (res[-1] < tol):
            break
    # tt += bsz
    return X[:, current_k % m].view_as(x0), res


def anderson_mse(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration.
    This was taken from the Deep Equilibrium tutorial here: http://implicit-layers-tutorial.org/deep_equilibrium_models/
    """

    # global tt
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape(x0.shape)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    recon_list = []
    current_k = 0
    past_iterate = x0
    for k in range(2, max_iter):
        current_k = k
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n) NEW

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape(x0.shape)).reshape(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        recon_list.append((X[:, k % m]).view_as(x0))

        if (res[-1] < tol):
            break
    # tt += bsz
    return X[:, current_k % m].view_as(x0), res, recon_list


class DEQIPFixedPoint(nn.Module):
    def __init__(self, invBlock, anderson, **kwargs):
        super(DEQIPFixedPoint, self).__init__()
        self.invBlock = invBlock
        self.solver = anderson
        self.kwargs = kwargs

    def forward(self, y, X0, Z0, A0, train=True):
        self.invBlock.init_setup(A0, Z0, X0)
        # by default, invblock does not require grad for updates in Xk
        Xk, forward_res = self.solver(lambda Xk: self.invBlock(Xk, y), X0, **self.kwargs)
        if train:
            # attach gradients
            Xk = self.invBlock(Xk, y, no_grad=False)
        return Xk


class DEQIPFixedPoint_v2(nn.Module):
    def __init__(self, invBlock, anderson, **kwargs):
        super(DEQIPFixedPoint_v2, self).__init__()
        self.invBlock = invBlock
        self.solver = anderson
        self.kwargs = kwargs

    def forward(self, y, X0, Z0, A0, train=True):
        self.invBlock.init_setup(A0, Z0, X0)
        # by default, invblock does not require grad for updates in Xk
        Xk, forward_res = self.solver(lambda Xk: self.invBlock(Xk, y), X0, **self.kwargs)
        Xk = self.invBlock(Xk, y, no_grad=False)

        if train:
            # setup Jacobian
            Xk0 = Xk.clone().detach()
            Xk0.requires_grad = True
            f0 = self.invBlock(Xk0, y, no_grad=False)

            def backward_hook(grad):
                g, self.backward_res = self.solver(lambda y: autograd.grad(f0, Xk0, y, retain_graph=True)[0] + grad,
                                                   grad, **self.kwargs)
                return g

            # attach gradients
            Xk.register_hook(backward_hook)
        return Xk


class DEQIPFixedPoint_v3(nn.Module):
    def __init__(self, invBlock, anderson, **kwargs):
        super(DEQIPFixedPoint_v3, self).__init__()
        self.invBlock = invBlock
        self.solver = anderson
        self.kwargs = kwargs

    def forward_iteration(self, X0, y):
        Xk = self.invBlock(X0, y)
        res = []
        Xk_list = []
        for k in range(self.kwargs['max_iter']):
            Xk_prev = Xk
            Xk = self.invBlock(Xk_prev, y, no_grad=True)
            res.append((Xk - Xk_prev).norm().item() / (1e-5 + Xk.norm().item()))
            Xk_list.append(Xk)
            if res[-1] < self.kwargs['tol']:
                break
        return Xk, Xk_list, res

    def forward(self, y, X0, Z0, A0, train=True):
        self.invBlock.init_setup(A0, Z0, X0)
        Xk, _, self.forward_res = self.forward_iteration(X0, y)
        if train:
            # attach gradients
            Xk = self.invBlock(Xk, y, no_grad=False)
        return Xk
