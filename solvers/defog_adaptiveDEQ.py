import sys, os, json, configargparse, collections, matplotlib, csv

sys.path.append("../")
from networks import network_hqs as nets
from utils.dataloader import *
from utils.misc import *
from operators import deq
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils.dataloader_defog import *
from pandas import *

matplotlib.use("Qt5Agg")
parser = configargparse.ArgParser()
parser.add_argument('--eta', type=float, default=-1, help='initial eta, lr for Xk')
parser.add_argument('--gamma', type=float, default=0.00, help='regularization coef for ||f_theta||')
parser.add_argument('--lamb', type=float, default=0.01, help='regularization coef for ||z-x||')
parser.add_argument('--shared_eta', action='store_true', help='Share eta across iterations K')
parser.add_argument('-c', '--my-config', is_config_file=True, help='config file path')
parser.add_argument("--location", type=str, default="coda", help="home or work directory")
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_A', type=float, default=1e-4, help='step size for updating A, 1e-5')
parser.add_argument('--lr_Z', type=float, default=1e-4, help='step size for updating Z')
parser.add_argument("--path", type=str, default="../saved_models/", help="network saving directory")
parser.add_argument("--load_path", type=str, default='../saved_models/')
parser.add_argument('--dataset', type=str, default="defog", help='CelebA, MRI, CT...')

parser.add_argument('--and_m', type=int, default=5, help='Anderson m')
parser.add_argument('--and_beta', type=float, default=1, help='Anderson beta')
parser.add_argument('--and_maxiters', type=int, default=30, help='Anderson max iters')
parser.add_argument('--and_tol', type=float, default=1e-3, help='Anderson tolerance, was 1e-3')

parser.add_argument("--file_name", type=str, default="2-defog/adaptiveDEQ/", help="saving directory")
parser.add_argument('--maxiters', type=int, default=5, help='Main max iterations')
parser.add_argument('--n_epochs', default=1)
parser.add_argument('--kernel_size', type=int, default=3, help='conv layer kernel size')
parser.add_argument('--padding', type=int, default=1, help='conv layer padding')
parser.add_argument('--batch_size', default=8, help='Batch size')
parser.add_argument('--batch_size_val', default=8, help='Batch size')
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
netA = nets.netA_fogging_sn(args).to(args.device)
netR = nets.proxNet_fog_sn(7, args).to(args.device)  # was 3 layers
optA = torch.optim.AdamW(netA.parameters(), lr=args.lr_A)
optR = torch.optim.AdamW(netR.parameters(), lr=args.lr)
criteria = nn.MSELoss()

""" DEQ SETUP """
invBlock = deq.defog_deq_kth_iter(args, netR, netA, optR, optA, criteria).to(args.device)
forward_iterator = deq.anderson
deq = deq.DEQIPFixedPoint_v3(invBlock, forward_iterator, m=args.and_m, beta=args.and_beta, lam=1e-3,
                          max_iter=args.and_maxiters, tol=args.and_tol)
opt = torch.optim.AdamW(deq.parameters(), lr=args.lr)
if args.pretrain:
    load_path = ''
    params = torch.load(load_path)['state_dict']
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
            for y, D, X in tr_loader:  # y, D, X = next(iter(tr_loader))
                bs = X.shape[0]
                y, D, X = y.to(args.device), D.to(args.device).type(torch.cuda.FloatTensor), X.to(args.device)
                X0, Z0 = torch.clone(y), torch.clone(y).to(args.device)

                Xk = deq(y, X0, Z0, D)
                tr_loss = criteria(Xk, X)
                opt.zero_grad()
                tr_loss.backward()
                opt.step()

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
                plot_foggy_X(X, Xk, y, args, epoch, 'defog adaptiveDEQ')
                plot_deq_residual(deq, D, Z0, y, X0, args, epoch)  # plot deq residual
                plot_deq_mse(deq, D, Z0, X0, X, y, criteria, epoch, args)

            for y, D, X in val_loader:
                bs = X.shape[0]
                y, D, X = y.to(args.device), D.to(args.device).type(torch.cuda.FloatTensor), X.to(args.device)
                X0, Z0 = torch.clone(y), torch.clone(y).to(args.device)

                Xk = deq(y, X0, Z0, D, train=False)
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

    plt.figure(figsize=(7,4))
    plt.semilogy(np.arange(len(tr_list)), tr_list)
    plt.semilogy(np.arange(len(val_list)), val_list)
    plt.legend(['train', 'val'])
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.tight_layout()
    plt.savefig(args.save_path + "/trajectory.png")

else:
    # initialize weight of netA with random variables
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.001)

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
        for y, D, X in ts_loader:
            bs = X.shape[0]
            y, D, X = y.to(args.device), D.to(args.device).type(torch.cuda.FloatTensor), X.to(args.device)
            X0, Z0 = torch.clone(y), torch.clone(y).to(args.device)

            # reset parameters in forward residual network
            deq.invBlock.netA.load_state_dict(netA_params)
            Xk = deq(y, X0, Z0, D, train=False)

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

        fh = open(trajectory_path, 'a', newline='')  # a for append
        csv_writer = csv.writer(fh)
        csv_writer.writerow([loss_meters[k].avg for k in range(len_meter)])
        fh.close()

    plot_foggy_X(X, Xk, y, args, 999, 'AdaptiveDEQ eval')
