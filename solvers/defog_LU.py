import matplotlib, csv, os, sys
sys.path.append("../")
from networks import network_lu as nets
from utils.dataloader import *
from utils.misc import *
import configargparse, json
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from utils.dataloader_defog import *
from pandas import *

matplotlib.use("Qt5Agg")
parser = configargparse.ArgParser()
parser.add_argument('--eta', type=float, default=-1, help='initial eta, lr for Xk')
parser.add_argument('--gamma', type=float, default=0.1, help='regularization coef for ||f_theta||')
parser.add_argument('--lamb', type=float, default=0.1, help='regularization coef for ||z-x||')
parser.add_argument('--shared_eta', action='store_true', help='Share eta across iterations K')
parser.add_argument('-c', '--my-config', is_config_file=True, help='config file path')
parser.add_argument("--location", type=str, default="home", help="home or work directory")
parser.add_argument('--lr', default=1e-4, help='learning rate')
parser.add_argument('--lr_A', default=2e-5, help='step size for updating A, 1e-5')
parser.add_argument('--lr_X', default=1e-4, help='step size for updating X')
parser.add_argument("--path", type=str, default="../saved_models/", help="network saving directory")
parser.add_argument("--load_path", type=str, default='../saved_models/')

parser.add_argument("--file_name", type=str, default="2-defog/lu/", help="saving directory")
parser.add_argument('--maxiters', type=int, default=5, help='Main max iterations')
parser.add_argument('--n_epochs', default=1)
parser.add_argument('--kernel_size', type=int, default=3, help='conv layer kernel size')
parser.add_argument('--padding', type=int, default=1, help='conv layer padding')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--batch_size_val', type=int, default=8, help='Batch size')
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
tr_loader, tr_length, val_loader, val_length, ts_loader, ts_length = load_foggy_data(args)

os.makedirs(args.save_path, exist_ok=True)
with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    print('joined successfully!')
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" NETWORK SETUP """
invBlock = nets.LU_deconv_defog(args, nlayers=7).to(args.device)
print("# Parmeters: ", sum(a.numel() for a in invBlock.parameters()))

opt = torch.optim.AdamW(invBlock.parameters(), lr=args.lr)
criteria = nn.MSELoss()

if args.pretrain:
    invBlock.load_state_dict(torch.load(args.load_path)['state_dict'])

""" BEGIN TRAINING """
if args.train:
    trajectory_path = args.save_path + '/trajectory.csv'
    fh = open(trajectory_path, 'a')
    csv_writer = csv.writer(fh)
    csv_writer.writerow(['train loss', 'val loss'])
    fh.close()

    for epoch in range(args.n_epochs):
        loss_meters = [AverageMeter() for _ in range(args.maxiters)]
        val_meter = AverageMeter()
        with tqdm(total=(tr_length - tr_length % args.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.n_epochs))
            for y, D, X in tr_loader:
                bs = y.shape[0]
                y, _, X = y.to(args.device), D.to(args.device).type(torch.cuda.FloatTensor), X.to(args.device)

                X0 = torch.clone(y)
                Xk = torch.clone(y)

                for k in range(args.maxiters):
                    Xk = invBlock(Xk, y)
                    loss_k = criteria(Xk.squeeze(), X.squeeze())
                    loss_meters[k].update(loss_k.item(), bs)
                loss = criteria(Xk.squeeze(), X.squeeze())
                opt.zero_grad()
                loss.backward()
                opt.step()

                torch.cuda.empty_cache()
                dict = {f'x{k}': f'{loss_meters[k].avg:.6f}' for k in range(args.maxiters)}
                dict.update({'ts_mse': f'{val_meter.avg:.6f}'})
                _tqdm.set_postfix(dict)
                _tqdm.update(bs)

            if (epoch + 1) % 10 == 0:
                plot_deblur(X, X0, Xk, args, epoch, algo="LU")
                state = {
                    'epoch': epoch,
                    'state_dict': invBlock.state_dict()}
                torch.save(state, os.path.join(args.save_path, f'epoch_{epoch}.state'))

            with torch.no_grad():
                for y, D, X in val_loader:
                    bs = y.shape[0]
                    y, _, X = y.to(args.device), D.to(args.device).type(torch.cuda.FloatTensor), X.to(args.device)

                    X0 = torch.clone(y)
                    Xk = torch.clone(y)

                    for k in range(args.maxiters):
                        Xk = invBlock(Xk, y)
                        loss_k = criteria(Xk.squeeze(), X.squeeze())
                    val_loss = criteria(Xk.squeeze(), X.squeeze())
                    val_meter.update(val_loss.item(), bs)

                    dict = {f'x{k}': f'{loss_meters[k].avg:.6f}' for k in range(args.maxiters)}
                    dict.update({'ts_mse': f'{val_meter.avg:.6f}'})
                    _tqdm.set_postfix(dict)
                    _tqdm.update(bs)

            fh = open(trajectory_path, 'a', newline='')  # a for append
            csv_writer = csv.writer(fh)
            csv_writer.writerow([loss_meters[-1].avg, val_meter.avg])
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
    ts_mse_meters = [AverageMeter() for _ in range(args.maxiters + 1)]

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
            with torch.no_grad():
                bs = y.shape[0]
                y, _, X = y.to(args.device), D.to(args.device).type(torch.cuda.FloatTensor), X.to(args.device)

                X0 = torch.clone(y)
                Xk = torch.clone(y)
                ts_mse_meters[0].update(criteria(X0, X).item(), bs)

                for k in range(args.maxiters):
                    Xk = invBlock(Xk, y)
                    torch.cuda.synchronize()
                    loss_k = criteria(Xk.squeeze(), X.squeeze())
                    ts_mse_meters[k + 1].update(loss_k.item(), bs)

                ts_loss = criteria(Xk, X)
                avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim = compute_metrics3chan(Xk, X, y)
                criteria_list = [ts_loss, avg_init_psnr, avg_recon_psnr, avg_delta_psnr, avg_ssim]
                for k in range(len_meter):
                    loss_meters[k].update(criteria_list[k].item(), 1)

                psnr_list.append(avg_recon_psnr.item())
                ssim_list.append(avg_ssim.item())

                torch.cuda.empty_cache()
                dict = {f'x{k}': f'{ts_mse_meters[k].avg:.6f}' for k in range(args.maxiters + 1)}
                dict.update({f'{criteria_title[k]}': f'{loss_meters[k].avg:.6f}' for k in range(len_meter)})
                _tqdm.set_postfix(dict)
                _tqdm.update(bs)

        fh = open(trajectory_path, 'a', newline='')  # a for append
        csv_writer = csv.writer(fh)
        csv_writer.writerow([loss_meters[k].avg for k in range(len_meter)])
        fh.close()
    plot_foggy_X(X, Xk, y, args, -1)
