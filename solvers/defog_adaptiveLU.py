import matplotlib, csv
import sys

sys.path.append("../")
from networks import network_hqs as nets
from utils.dataloader import *
from utils.misc import *
import configargparse
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm
from utils.dataloader_defog import *
from pandas import *

matplotlib.use("Qt5Agg")
parser = configargparse.ArgParser()
parser.add_argument('--eta', type=float, default=-5, help='initial eta, lr for Xk')
parser.add_argument('--gamma', type=float, default=0.0001, help='regularization coef for ||f_theta||')
parser.add_argument('--lamb', type=float, default=0.1, help='regularization coef for ||z-x||')
parser.add_argument('--shared_eta', action='store_true', help='Share eta across iterations K')
parser.add_argument('-c', '--my-config', is_config_file=True, help='config file path')
parser.add_argument("--location", type=str, default="home", help="home or work directory")
parser.add_argument('--lr', default=1e-4, help='learning rate')
parser.add_argument('--lr_A', default=1e-4, help='step size for updating A, 1e-5')
parser.add_argument('--lr_X', default=1e-4, help='step size for updating X')
parser.add_argument("--path", type=str, default="../saved_models/", help="network saving directory")
parser.add_argument("--load_path", type=str, default='../saved_models/')

parser.add_argument("--file_name", type=str, default="2-defog/adaptiveLU/", help="saving directory")
parser.add_argument('--maxiters', type=int, default=5, help='Main max iterations')
parser.add_argument('--n_epochs', default=1)
parser.add_argument('--kernel_size', type=int, default=3, help='conv layer kernel size')
parser.add_argument('--padding', type=int, default=1, help='conv layer padding')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--batch_size_val', type=int, default=8, help='Batch size')
parser.add_argument('--pretrain', type=bool, default=True, help='if load utils and resume training')
parser.add_argument('--train', type=bool, default=False, help='training or eval mode')

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
tr_loader, tr_length, val_loader, val_length, ts_loader, ts_length = load_foggy_data(args)

os.makedirs(args.save_path, exist_ok=True)
with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    print('joined successfully!')
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" NETWORK SETUP """
netA = nets.netA_foggy(args).to(args.device)
netR = nets.proxNet_fog(7, args).to(args.device)
optA = torch.optim.AdamW(netA.parameters(), lr=args.lr_A)
optR = torch.optim.AdamW(netR.parameters(), lr=args.lr)
criteria = nn.MSELoss()

if args.pretrain:
    load_path_A = ''
    load_path_R = ''
    netA.load_state_dict(torch.load(load_path_A)['state_dict'])
    netR.load_state_dict(torch.load(load_path_R)['state_dict'])
    print('Model loaded successfully!')


""" BEGIN TRAINING """
if args.train:
    trajectory_path = args.save_path + '/trajectory.csv'
    fh = open(trajectory_path, 'a')
    csv_writer = csv.writer(fh)
    csv_writer.writerow(['tr_A loss', 'tr_R loss', 'val_A loss', 'val_R loss'])
    fh.close()

    for epoch in range(args.n_epochs):
        loss_meters = [AverageMeter() for _ in range(args.maxiters)]
        A_meters = AverageMeter()
        A_meters_val = AverageMeter()
        val_meter = AverageMeter()
        with tqdm(total=(tr_length - tr_length % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.n_epochs))
            for y, D, X in tr_loader:  # y, D, X = next(iter(tr_loader))
                bs = y.shape[0]
                y, D, X = y.to(args.device), D.to(args.device).type(torch.cuda.FloatTensor), X.to(args.device)

                X0 = torch.clone(y)
                Zk = torch.clone(X0)
                Xk = torch.clone(X0)

                for k in range(args.maxiters):
                    """ UPDATE X  plt.figure();plt.imshow(Xk[0,0].cpu()) """
                    netA.eval()
                    netR.eval()
                    optX = torch.optim.AdamW([Xk], lr=args.lr_X)
                    Xk.requires_grad_(True)
                    yinit, yk = netA(Xk, D)
                    optX.zero_grad()
                    lossX = criteria(y, yk) + args.gamma * criteria(yk, yinit) + args.lamb * criteria(Zk.detach(), Xk)
                    lossX.backward()
                    optX.step()
                    Xk = Xk.detach()

                    """ UPDATE netA """
                    netA.train()
                    netR.eval()
                    Xk.requires_grad_(False)
                    yinit, yk = netA(Xk, D)
                    optA.zero_grad()
                    lossA = criteria(y, yk) + args.gamma * criteria(yk, yinit)
                    A_meters.update(criteria(y, yk).item(), bs)
                    lossA.backward()
                    optA.step()

                    """ UPDATE z """
                    netA.eval()
                    netR.train()
                    Zk = netR(Zk, Xk, y)
                    lossR = criteria(Zk, X)
                    loss_meters[k].update(lossR.item(), bs)

                optR.zero_grad()
                lossR.backward()
                optR.step()

                torch.cuda.empty_cache()
                dict = {f'x{k}': f'{loss_meters[k].avg:.6f}' for k in range(args.maxiters)}
                dict.update({'lossA': f'{A_meters.avg:.6f}'})
                dict.update({'lossA_val': f'{A_meters_val.avg:.6f}'})
                dict.update({'recon_val': f'{val_meter.avg:.6f}'})
                _tqdm.set_postfix(dict)
                _tqdm.update(bs)

            plot_foggy_X(X, Zk, y, args, epoch)

            if (epoch + 1) % 5 == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': netA.state_dict()}
                torch.save(state, os.path.join(args.save_path, f'netA_epoch_{epoch}.state'))
                state = {
                    'epoch': epoch,
                    'state_dict': netR.state_dict()}
                torch.save(state, os.path.join(args.save_path, f'netR_epoch_{epoch}.state'))

            for y, D, X in val_loader:
                bs = y.shape[0]
                y, D, X = y.to(args.device), D.to(args.device).type(torch.cuda.FloatTensor), X.to(args.device)

                X0 = torch.clone(y)
                Zk = torch.clone(X0)
                Xk = torch.clone(X0)

                for k in range(args.maxiters):
                    # UPDATE X
                    netA.eval()
                    netR.eval()
                    optX = torch.optim.AdamW([Xk], lr=args.lr_X)
                    Xk.requires_grad_(True)
                    yinit, yk = netA(Xk, D)
                    optX.zero_grad()
                    lossX = criteria(y, yk) + args.gamma * criteria(yk, yinit) + args.lamb * criteria(Zk.detach(), Xk)
                    lossX.backward()
                    optX.step()
                    Xk = Xk.detach()

                    # UPDATE netA
                    netA.train()
                    netR.eval()
                    Xk.requires_grad_(False)
                    yinit, yk = netA(Xk, D)
                    optA.zero_grad()
                    lossA = criteria(y, yk) + args.gamma * criteria(yk, yinit)
                    A_meters_val.update(criteria(y, yk).item(), bs)
                    lossA.backward()
                    optA.step()

                    with torch.no_grad():
                        # UPDATE z
                        netA.eval()
                        netR.train()
                        Zk = netR(Zk, Xk, y)
                        lossR = criteria(Zk, X)

                val_loss = criteria(Zk, X)
                val_meter.update(val_loss.item(), bs)

                dict = {f'x{k}': f'{loss_meters[k].avg:.6f}' for k in range(args.maxiters)}
                dict.update({'lossA': f'{A_meters.avg:.6f}'})
                dict.update({'lossA_val': f'{A_meters_val.avg:.6f}'})
                dict.update({'recon_val': f'{val_meter.avg:.6f}'})
                _tqdm.set_postfix(dict)
                _tqdm.update(bs)

            fh = open(trajectory_path, 'a', newline='')  # a for append
            csv_writer = csv.writer(fh)
            csv_writer.writerow([A_meters.avg, loss_meters[-1].avg, A_meters_val.avg, val_meter.avg])
            fh.close()

    """ READ TRAJECTORY """
    traj = read_csv(trajectory_path)
    trA_list = traj['tr_A loss'].tolist()
    trR_list = traj['tr_R loss'].tolist()
    valA_list = traj['val_A loss'].tolist()
    valR_list = traj['val_R loss'].tolist()
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(trA_list)), trA_list)
    plt.plot(np.arange(len(valA_list)), valA_list)
    plt.title('loss A')
    plt.legend(['train', 'val'])
    # plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(trR_list)), trR_list)
    plt.plot(np.arange(len(valR_list)), valR_list)
    plt.title('loss R')
    plt.legend(['train', 'val'])
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig(args.save_path + '/trajectory.png')

else:
    criteria = nn.MSELoss()
    criteria_title = ['mse', 'avgInit', 'avgPSNR', 'deltaPSNR', 'avgSSIM', 'runtime']
    len_meter = len(criteria_title)
    loss_meters = [AverageMeter() for _ in range(len_meter)]
    ts_mse_meters = [AverageMeter() for _ in range(args.maxiters + 1)]
    A0_meters = AverageMeter()
    A_meters_1iter = AverageMeter()
    A_meters_Kiters = AverageMeter()

    trajectory_path = args.save_path + '/trajectory_test.csv'
    fh = open(trajectory_path, 'a')
    csv_writer = csv.writer(fh)
    csv_writer.writerow(criteria_title)
    fh.close()

    psnr_list = []
    ssim_list = []
    with tqdm(total=(ts_length - ts_length % args.batch_size)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(1, 1))
        for y, D, X in ts_loader:
            bs = y.shape[0]
            y, D, X = y.to(args.device), D.to(args.device).type(torch.cuda.FloatTensor), X.to(args.device)

            X0 = torch.clone(y)
            Zk = torch.clone(X0)
            Xk = torch.clone(X0)
            ts_mse_meters[0].update(criteria(X0, X).item(), bs)
            A0_meters.update(criteria(y, X).item(), bs)

            for k in range(args.maxiters):
                # UPDATE X
                netA.eval()
                netR.eval()
                optX = torch.optim.AdamW([Xk], lr=args.lr_X)
                Xk.requires_grad_(True)
                yinit, yk = netA(Xk, D)
                optX.zero_grad()
                lossX = criteria(y, yk) + args.gamma * criteria(yk, yinit) + args.lamb * criteria(Zk.detach(), Xk)
                lossX.backward()
                optX.step()
                Xk = Xk.detach()

                # UPDATE netA
                netA.train()
                netR.eval()
                Xk.requires_grad_(False)
                yinit, yk = netA(Xk, D)
                optA.zero_grad()
                lossA = criteria(y, yk) + args.gamma * criteria(yk, yinit)
                if k == 0:
                    A_meters_1iter.update(criteria(y, yk).item(), bs)
                elif k == args.maxiters - 1:
                    A_meters_Kiters.update(criteria(y, yk).item(), bs)
                lossA.backward()
                optA.step()

                with torch.no_grad():
                    # UPDATE z
                    netA.eval()
                    netR.train()
                    Zk = netR(Zk, Xk, y)
                    lossR = criteria(Zk, X)
                    ts_mse_meters[k + 1].update(lossR.item(), bs)

            ts_loss = criteria(Zk, X)
            avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim = compute_metrics3chan(Zk, X, X0)
            criteria_list = [ts_loss, avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim]
            for k in range(len_meter):
                loss_meters[k].update(criteria_list[k].item(), bs)

            psnr_list.append(avg_recon_psnr.item())
            ssim_list.append(avg_ssim.item())

            torch.cuda.empty_cache()
            dict = {f'x{k}': f'{ts_mse_meters[k].avg:.6f}' for k in range(args.maxiters + 1)}
            dict.update({f'{criteria_title[k]}': f'{loss_meters[k].avg:.6f}' for k in range(len_meter)})
            dict.update({f'A0': f'{A0_meters.avg:.6f}', f'A_1iter': f'{A_meters_1iter.avg:.6f}', f'A_Kiter': f'{A_meters_Kiters.avg:.6f}'})

            _tqdm.set_postfix(dict)
            _tqdm.update(bs)

        fh = open(trajectory_path, 'a', newline='')  # a for append
        csv_writer = csv.writer(fh)
        csv_writer.writerow([loss_meters[k].avg for k in range(len_meter)])
        fh.close()
    plot_foggy_X(X, Zk, y, args, -1)
