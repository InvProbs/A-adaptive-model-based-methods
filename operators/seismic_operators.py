import torch
from scipy.linalg import convolution_matrix


class LinearOperator(torch.nn.Module):
    def __init__(self):
        super(LinearOperator, self).__init__()

    def forward(self, x):
        pass

    def adjoint(self, x):
        pass

    def gramian(self, x):
        return self.adjoint(self.forward(x))


class SelfAdjointLinearOperator(LinearOperator):
    def adjoint(self, x):
        return self.forward(x)


class Identity(SelfAdjointLinearOperator):
    def forward(self, x):
        return x


class OperatorPlusNoise(torch.nn.Module):
    def __init__(self, operator, noise_sigma):
        super(OperatorPlusNoise, self).__init__()
        self.internal_operator = operator
        self.noise_sigma = noise_sigma

    def forward(self, x):
        A_x = self.internal_operator(x)
        return A_x + self.noise_sigma * torch.randn_like(A_x)


def normalize(X, bs):
    maxVal, _ = torch.max(X.reshape(bs, -1), dim=1)
    minVal, _ = torch.min(X.reshape(bs, -1), dim=1)
    return (X - minVal[:, None, None]) / (maxVal - minVal)[:, None, None]


class convolution(LinearOperator):
    def __init__(self, W, trace_length):
        super(convolution, self).__init__()
        self.A = W  # Wavelet Matrix
        self.trace_len = trace_length

    def forward(self, x):
        return self.A @ x

    def gramian(self, x):
        return self.A.T @ self.A @ x

    def adjoint(self, y):
        return self.A.T @ y


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration.
    This was taken from the Deep Equilibrium tutorial here: http://implicit-layers-tutorial.org/deep_equilibrium_models/
    """

    bsz, L = x0.shape
    d = 1
    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
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
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        current_iterate = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape(x0.shape)).reshape(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))

        if (res[-1] < tol):
            break
    # tt += bsz
    return X[:, current_k % m].view_as(x0), res


def anderson4(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
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
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n) NEW
        # alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0] # (bsz x n) ORIGINAL

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        current_iterate = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape(x0.shape)).reshape(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))

        # plt.plot((X[:, current_k % m].view_as(x0) + dk)[0].squeeze().detach().cpu())
        # anderson4(lambda xk: invBlock.R.g(inj, xk), dk, m=5, beta=3, max_iter=50)

        if (res[-1] < tol):
            break
    # tt += bsz
    return X[:, current_k % m].view_as(x0), res
