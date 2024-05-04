import math, torch, os
import torch.nn as nn
import matplotlib.pyplot as plt
# import pytorch_ssim as pytorch_ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from operators import deq as deq_op


def normalize(yn, X, bs):
    maxVal, _ = torch.max(torch.abs(yn.reshape(bs, -1)), dim=1)
    # maxVal[maxVal < 0.1] = 1
    if len(X.shape) == 3:
        return maxVal, yn / maxVal[:, None, None, None], X / maxVal[:, None, None]
    return maxVal, yn / maxVal[:, None, None, None], X / maxVal[:, None, None, None]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def PSNR1chan(Xk, X):  # ONLY the REAL Part
    bs, C, W, H = X.shape
    Xk = Xk[:, 0, :, :]
    X = X[:, 0, :, :]
    mse = torch.sum(((Xk - X) ** 2).reshape(bs, -1), dim=1) / (W * H)
    return 20 * torch.log10(torch.max(torch.max(X, dim=1)[0], dim=1)[0] / torch.sqrt(mse))


def compute_metrics1chan(Xk, X, X0):
    init_psnr, recon_psnr = PSNR1chan(X0, X), PSNR1chan(Xk, X)
    bs = X.shape[0]
    avg_init_psnr = torch.sum(init_psnr) / bs
    avg_recon_psnr = torch.sum(recon_psnr) / bs
    avg_delta_psnr = torch.sum(recon_psnr - init_psnr) / bs

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    avg_ssim = ssim(Xk.cpu(), X.cpu())
    return avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim


def PSNR3chan(Xk, X):
    bs, C, W, H = X.shape
    mse = torch.sum(((Xk - X) ** 2).reshape(bs, -1), dim=1) / (C * W * H)
    return 20 * torch.log10(1 / torch.sqrt(mse))


def compute_metrics3chan(Xk, X, X0):
    init_psnr, recon_psnr = PSNR3chan(X0, X), PSNR3chan(Xk, X)
    bs = X.shape[0]
    avg_init_psnr = torch.sum(init_psnr) / bs
    avg_recon_psnr = torch.sum(recon_psnr) / bs
    avg_delta_psnr = torch.sum(recon_psnr - init_psnr) / bs
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    avg_ssim = ssim(Xk.cpu(), X.cpu())
    return avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim


def plot_reflectivity(X, X0, Xk, args, epoch):
    plt.figure(figsize=(7, 5))
    plt.suptitle('1D Loop Unrolling results')
    Xk = Xk.detach().cpu().squeeze()
    X = X.detach().cpu().squeeze()
    X0 = X0.detach().cpu().squeeze()
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(X[i, :, 25], '-r')
        plt.plot(Xk[i, :, 25], '--b')
        # plt.plot(X0[i, :, 25], '-g')
        plt.legend(['true reflectivity', 'reconstructed reflectivity'])
        # plt.legend(['reconstructed reflectivity', 'true reflectivity', 'trace'])

    if args.train:
        plt.savefig(os.path.join(args.save_path, f'{epoch}_result.png'))
    else:
        plt.savefig(os.path.join(args.save_path, f'test_{epoch}_result.png'))
    plt.close()

    # plot 2D
    i = 1
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(X0[i], cmap='gray')
    plt.title('Observed Trace')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(Xk[i], cmap='gray')
    plt.title('Recovered Reflectivity')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(X[i], cmap='gray')
    plt.title('Ground Truth Reflectivity')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, f'2D_epoch_{epoch}.png'))
    plt.close()


def plot_deq_residual(deq, A0, Z0, y, X0, args, epoch=0):
    """ evaluate intermediate deq results """
    deq.invBlock.init_setup(A0, Z0, X0)
    xk, forward_res = deq.solver(lambda xk: deq.invBlock(xk, y), X0, **deq.kwargs)
    # xk, forward_res = deq.solver(lambda xk: deq.invBlock(xk, y), torch.zeros_like(X0), **deq.kwargs)
    plt.figure(figsize=(7, 3))
    plt.semilogy(forward_res)
    plt.xlabel('DEQ iterations')
    plt.ylabel('log(residual)')
    plt.title('DEQ residual plot')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, f'deq_res_{epoch}.png'))
    plt.close()


def plot_deq_mse(deq, A0, Z0, X0, X, y, criteria, epoch, args):
    """ evaluate intermediate deq results """
    deq.invBlock.init_setup(A0, Z0, X0)
    # xk, res, intermediate_xk = deq_op.anderson_mse(lambda xk: deq.invBlock(xk, y), X0, **deq.kwargs)
    xk, xk_list, res = deq.forward_iteration(X0, y)
    mse_list = []
    psnr_list = []
    for xk in xk_list:
        mse_list.append(criteria(xk, X).item())
        _, psnr, _, _ = compute_metrics3chan(xk, X, y)
        psnr_list.append(psnr.item())
    plt.figure(figsize=(7, 3))
    plt.semilogy(mse_list)
    plt.xlabel('DEQ iterations')
    plt.ylabel('MSE')
    plt.title('Intermediate reconstruction MSE of DEQ iterations')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_path, f'{epoch}_DEQ_mse_result.png'))
    plt.close()

def plot_foggy_X(X, Xk, y, args, epoch, title=''):
    plt.figure(figsize=(5, 7))
    plt.suptitle(title)
    X = torch.clamp(X.detach().cpu().permute(0, 2, 3, 1), 0, 1)
    Xk = torch.clamp(Xk.detach().cpu().permute(0, 2, 3, 1), 0, 1)
    y = torch.clamp(y.detach().cpu().permute(0, 2, 3, 1), 0, 1)
    i = 3
    plt.subplot(3, 1, 1)
    plt.imshow(y[i])
    plt.axis('off')
    plt.title('$X_0$')
    plt.subplot(3, 1, 2)
    plt.imshow(Xk[i])
    plt.axis('off')
    plt.title('$\hat{X}$')
    plt.subplot(3, 1, 3)
    plt.imshow(X[i])
    plt.axis('off')
    plt.title('$X$')
    if args.train:
        plt.savefig(os.path.join(args.save_path, f'{epoch}_result.png'))
    else:
        plt.savefig(os.path.join(args.save_path, f'test_{epoch}_result.png'))
    plt.close()

def plot_deblur(X, X0, Xk, args, epoch, algo="Adaptive LU"):
    plt.figure(figsize=(4, 6))
    plt.suptitle('Deblurring ' + algo)
    Xk = torch.clamp(Xk, 0, 1).detach().cpu().squeeze()
    X = torch.clamp(X, 0, 1).detach().cpu().squeeze()
    X0 = torch.clamp(X0, 0, 1).detach().cpu().squeeze()
    index = 0
    for i in range(3):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X0[index + i].permute(1, 2, 0))
        plt.title('$X_0$')
        plt.axis('off')
        plt.subplot(3, 3, i + 4)
        plt.imshow(Xk[index +i].permute(1, 2, 0))
        plt.title('$\hat{X}$')
        plt.axis('off')
        plt.subplot(3, 3, i + 7)
        plt.imshow(X[index +i].permute(1, 2, 0))
        plt.title('$X$')
        plt.axis('off')
    plt.tight_layout()

    if args.train:
        plt.savefig(os.path.join(args.save_path, f'{epoch}_result.png'))
    else:
        plt.savefig(os.path.join(args.save_path, f'test_{epoch}_result.png'))
    plt.close()
