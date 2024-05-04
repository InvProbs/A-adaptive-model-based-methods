import collections

import matplotlib, csv, collections
import matplotlib.pyplot as plt
import torch.optim

from networks import network_hqs as nets
from operators import deq
import torch.optim as optim
from utils.dataloader import *
from utils.misc import *
import configargparse
import os
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm
# from utils.seis_utils import *
from utils.dataloader_seisinv import *
from pandas import *
from utils.forward_models import GaussianBlur

matplotlib.use("Qt5Agg")
parser = configargparse.ArgParser()
parser.add_argument('--eta', type=float, default=-1, help='initial eta, lr for Xk')
parser.add_argument('--gamma', type=float, default=0.01, help='regularization coef for ||f_theta||')
parser.add_argument('--lamb', type=float, default=0.1, help='regularization coef for ||z-x||')
parser.add_argument('--shared_eta', action='store_true', help='Share eta across iterations K')
parser.add_argument('-c', '--my-config', is_config_file=True, help='config file path')
parser.add_argument("--location", type=str, default="coda", help="home or work directory")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_A', type=float, default=1e-4, help='step size for updating A, 1e-5')
parser.add_argument('--lr_Z', type=float, default=1e-4, help='step size for updating Z')
parser.add_argument("--path", type=str, default="../saved_models/", help="network saving directory")
parser.add_argument("--load_path", type=str, default='../saved_models/')
parser.add_argument('--dataset', type=str, default="CelebA", help='CelebA, MRI, CT...')

parser.add_argument('--and_m', type=int, default=5, help='Anderson m')
parser.add_argument('--and_beta', type=float, default=1, help='Anderson beta')
parser.add_argument('--and_maxiters', type=int, default=40, help='Anderson max iters')
parser.add_argument('--and_tol', type=float, default=1e-3, help='Anderson tolerance, was 1e-3')

parser.add_argument("--file_name", type=str, default="4-deblur/adaptiveDEQ/", help="saving directory")
parser.add_argument('--maxiters', type=int, default=5, help='Main max iterations')
parser.add_argument('--n_epochs', default=1)
parser.add_argument('--kernel_size', type=int, default=3, help='conv layer kernel size')
parser.add_argument('--padding', type=int, default=1, help='conv layer padding')
parser.add_argument('--batch_size', default=32, help='Batch size')
parser.add_argument('--batch_size_val', default=32, help='Batch size')
parser.add_argument('--pretrain', type=bool, default=False, help='if load utils and resume training')
parser.add_argument('--train', type=bool, default=True, help='training or eval mode')

parser.add_argument('--train_identity', type=bool, default=False)
parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
parser.add_argument('--noise_level', type=float, default=0.01)
parser.add_argument('--nc', type=int, default=1, help='number of channels in an image')
parser.add_argument('--lr_gamma', type=float, default=0.5)
parser.add_argument('--sched_step', type=int, default=300)
args = parser.parse_args()
args.shared_eta = True
print(args)
cuda = True if torch.cuda.is_available() else False
print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
cudnn.benchmark = True
random.seed(110)
torch.manual_seed(110)

""" LOAD DATA """
tr_loader, tr_length, val_loader, val_length, ts_loader, ts_length, args.save_path = load_data(args)

os.makedirs(args.save_path, exist_ok=True)
with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    print('joined successfully!')
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" NETWORK SETUP """
netA = nets.netA_deblur_sn(args, nlayers=3).to(args.device)
netR = nets.proxNet_deblur_sn(args, nlayers=5).to(args.device)
optA = torch.optim.AdamW(netA.parameters(), lr=args.lr_A)
optR = torch.optim.AdamW(netR.parameters(), lr=args.lr)
criteria = nn.MSELoss()

""" DEQ SETUP """
invBlock = deq.deblur_deq_kth_iter(args, netR, netA, optR, optA, criteria).to(args.device)
forward_iterator = deq.anderson
deq = deq.DEQIPFixedPoint(invBlock, forward_iterator, m=args.and_m, beta=args.and_beta, lam=1e-3,
                                        max_iter=args.and_maxiters, tol=args.and_tol)
opt = torch.optim.AdamW(deq.parameters(), lr=args.lr)
if args.pretrain:
    params = torch.load(args.load_path)['state_dict']
    del params['invBlock.A0.gaussian_kernel']
    deq.load_state_dict(params)
    print('Model loaded successfully!')

    netA_params = collections.OrderedDict()
    for key, param in params.items():
        if 'netA' in key:
            netA_params[key[14:]] = param
    deq.invBlock.netA.load_state_dict(netA_params)

""" BEGIN TRAINING """
if args.train:
    trajectory_path = args.save_path + '/trajectory.csv'
    fh = open(trajectory_path, 'a')
    csv_writer = csv.writer(fh)
    csv_writer.writerow(['train loss', 'val loss'])
    fh.close()

    for epoch in range(args.n_epochs):
        loss_meters = AverageMeter()
        val_meters = AverageMeter()
        with tqdm(total=(tr_length - tr_length % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.n_epochs))
            for X, _ in tr_loader:
                bs = X.shape[0]
                # measurement from true forward model
                addBlur = GaussianBlur(sigma=np.random.uniform(5, 11), kernel_size=np.random.randint(1, 5) * 2 + 1,
                                       n_channels=3).to(args.device)
                X = X.to(args.device)
                y = addBlur(X).to(args.device)

                # setup estimated forward model
                A0 = GaussianBlur(sigma=7, kernel_size=7, n_channels=3).to(args.device)
                X0, Z0 = torch.clone(y), torch.clone(y)

                Xk = deq(y, X0, Z0, A0)
                tr_loss = criteria(Xk, X)
                opt.zero_grad()
                tr_loss.backward()
                opt.step()

                loss_meters.update(tr_loss.item(), bs)
                torch.cuda.empty_cache()
                dict = {f'tr_mse': f'{loss_meters.avg:.6f}'}
                dict.update({'ts_mse': f'{val_meters.avg:.6f}'})
                _tqdm.set_postfix(dict)
                _tqdm.update(bs)

            """ SAVE STATES """
            if (epoch + 1) % 1 == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': deq.state_dict(),
                    'optimizer': opt.state_dict()
                }
                torch.save(state, os.path.join(args.save_path, f'epoch_{epoch}.state'))
                plot_deblur(X, X0, Xk, args, epoch, algo="HQS-DEQ")
                plot_deq_residual(deq, A0, Z0, y, X0, args, epoch) # plot deq residual

            for X, _ in val_loader:
                bs = X.shape[0]
                # measurement from true forward model
                addBlur = GaussianBlur(sigma=np.random.uniform(5, 11), kernel_size=np.random.randint(1, 5) * 2 + 1,
                                       n_channels=3).to(args.device)
                X = X.to(args.device)
                y = addBlur(X).to(args.device)
                A0 = GaussianBlur(sigma=7, kernel_size=7, n_channels=3).to(args.device)
                X0, Z0 = torch.clone(y), torch.clone(y)
                # deq.invBlock.netA.load_state_dict(netA_params)

                Xk = deq(y, X0, Z0, A0, train=False)
                val_loss = criteria(Xk, X)

                val_meters.update(val_loss.item(), bs)
                torch.cuda.empty_cache()
                dict = {f'tr_mse': f'{loss_meters.avg:.6f}'}
                dict.update({'ts_mse': f'{val_meters.avg:.6f}'})
                _tqdm.set_postfix(dict)
                _tqdm.update(bs)

            fh = open(trajectory_path, 'a', newline='')  # a for append
            csv_writer = csv.writer(fh)
            csv_writer.writerow([loss_meters.avg, val_meters.avg])
            fh.close()

    """ READ TRAJECTORY """
    traj = read_csv(trajectory_path)
    tr_list = traj["train loss"].tolist()
    val_list = traj["val loss"].tolist()

    plt.figure()
    plt.semilogy(np.arange(len(tr_list)), tr_list)
    plt.semilogy(np.arange(len(val_list)), val_list)
    plt.legend(['train', 'val'])
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig(args.save_path + "/trajectory.png")

else:
    criteria = nn.MSELoss()
    criteria_title = ['mse', 'avgInit', 'avgPSNR', 'deltaPSNR', 'avgSSIM']
    len_meter = len(criteria_title)
    loss_meters = [AverageMeter() for _ in range(len_meter)]
    ts_mse_meters = AverageMeter()

    trajectory_path = args.save_path + '/trajectory_test.csv'
    fh = open(trajectory_path, 'a')
    csv_writer = csv.writer(fh)
    csv_writer.writerow(criteria_title)
    fh.close()

    psnr_list = []
    ssim_list = []
    with tqdm(total=(ts_length - ts_length % args.batch_size)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(1, 1))
        for X, _ in ts_loader:
            bs = X.shape[0]
            # measurement from true forward model
            addBlur = GaussianBlur(sigma=np.random.uniform(5, 11), kernel_size=np.random.randint(1, 5) * 2 + 1,
                                   n_channels=3).to(args.device)
            X = X.to(args.device)
            y = addBlur(X).to(args.device)
            A0 = GaussianBlur(sigma=7, kernel_size=7, n_channels=3).to(args.device)
            X0, Z0 = torch.clone(y), torch.clone(y)

            # reset parameters in forward residual network
            deq.invBlock.netA.load_state_dict(netA_params)

            Xk = deq(y, X0, Z0, A0, train=False)
            ts_loss = criteria(Xk, X)
            ts_mse_meters.update(ts_loss.item(), bs)
            avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim = compute_metrics3chan(Xk, X, y)
            criteria_list = [ts_loss, avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim]
            for k in range(len_meter):
                loss_meters[k].update(criteria_list[k].item(), 1)

            psnr_list.append(avg_recon_psnr.item())
            ssim_list.append(avg_ssim.item())

            torch.cuda.empty_cache()
            dict = {f'{criteria_title[k]}': f'{loss_meters[k].avg:.6f}' for k in range(len_meter)}
            _tqdm.set_postfix(dict)
            _tqdm.update(bs)
            # break
        fh = open(trajectory_path, 'a', newline='')  # a for append
        csv_writer = csv.writer(fh)
        csv_writer.writerow([loss_meters[k].avg for k in range(len_meter)])
        fh.close()

    plot_deblur(X, X0, Xk, args, -1, algo="Adaptive DEQ")
