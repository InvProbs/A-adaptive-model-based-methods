import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.fftpack import fft, dct
from scipy import signal
import torch
import scipy.io
import scipy, mat73, os, datetime
from torch.utils.data import Dataset

""" FOR SEISMIC DECONVOLUTION PROBLEM"""


def readFile2D():
    true_wavelet = mat73.loadmat('../data/reflectivity_test_1D.mat')['W']
    W = scipy.linalg.convolution_matrix(true_wavelet.squeeze(), 352)
    return true_wavelet, W[25:377, :]


class Custom2DDataset(Dataset):
    def __init__(self, data_dir, refl_transform=None, trace_transform=None):
        self.root_dir = data_dir
        self.refl_transform = refl_transform
        self.trace_transform = trace_transform
        self.files = os.listdir(data_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        ntraces = 50
        trace_length = 128
        trace = scipy.io.loadmat(path)['NoisySig']
        refl = scipy.io.loadmat(path)['syn_model']
        W = scipy.io.loadmat(path)['W']
        # trueA = scipy.io.loadmat(path)['trueA']
        if self.refl_transform:
            refl = self.refl_transform(refl)
        if self.trace_transform:
            trace = self.trace_transform(trace)
        W = self.refl_transform(W)
        # trueA = self.refl_transform(trueA)
        return trace, refl, W  # , trueA


def gen_dataloader2D(args):
    train_path = '../data/learnA_uncertain_kernel_train'
    val_path = '../data/learnA_uncertain_kernel_val/'
    test_path = '../data/learnA_uncertain_kernel_test/'
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = Custom2DDataset(train_path, refl_transform=transform, trace_transform=transform)
    val_dataset = Custom2DDataset(val_path, refl_transform=transform, trace_transform=transform)
    test_dataset = Custom2DDataset(test_path, refl_transform=transform, trace_transform=transform)
    tr_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size_val, shuffle=True, drop_last=False)
    ts_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)

    timeStamp = datetime.datetime.now().strftime("%m%d-%H%M")
    args.save_path = args.path + args.file_name + timeStamp
    args.save_path = args.save_path + '_ts' if not args.train else args.save_path
    return tr_loader, len(train_dataset), val_loader, len(val_dataset), ts_loader, len(test_dataset)


""" DATALOADER FOR SEISMIC INVERSION PROBLEM """


class CustomSeisInvDataset(Dataset):
    def __init__(self, data_path, refl_transform=None, trace_transform=None, rtm_transform=None):
        self.trace_dir = os.path.join(data_path, 'trace_3shots/')
        self.refl_dir = os.path.join(data_path, 'refl/')
        self.rtm_dir = os.path.join(data_path, 'rtm/')
        self.trace_files = os.listdir(self.trace_dir)
        self.refl_files = sorted(os.listdir(self.refl_dir))
        self.rtm_files = sorted(os.listdir(self.rtm_dir))

        self.refl_transform = refl_transform
        self.trace_transform = trace_transform
        self.rtm_transform = rtm_transform

    def __len__(self):
        return len(self.refl_files)

    def __getitem__(self, idx):
        trace_path = os.path.join(self.trace_dir, self.trace_files[idx])
        refl_path = os.path.join(self.refl_dir, self.refl_files[idx])
        rtm_path = os.path.join(self.rtm_dir, self.rtm_files[idx])
        # rtm_path = '//wsl.localhost/Ubuntu-20.04/home/guanp/dataset/train/init/init_010.npy'
        trace = np.load(trace_path)[1, :, :99]  # (1001, 99)
        refl = np.load(refl_path)  # (341, 318)
        rtm = np.load(rtm_path) / 1000  # (341, 318)
        if self.refl_transform:
            refl = self.refl_transform(refl)
        if self.trace_transform:
            trace = self.trace_transform(trace)
        if self.rtm_transform:
            rtm = self.rtm_transform(rtm)
        return trace, refl, rtm


def gen_seisinv_dataloader(args):
    train_path = 'D:/seis_inv_dataset/train/'
    test_path = 'D:/seis_inv_dataset/test/'

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CustomSeisInvDataset(train_path, refl_transform=transform, trace_transform=transform,
                                         rtm_transform=transform)
    test_dataset = CustomSeisInvDataset(test_path, refl_transform=transform, trace_transform=transform,
                                        rtm_transform=transform)
    tr_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    ts_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)

    timeStamp = datetime.datetime.now().strftime("%m%d-%H%M")
    args.save_path = args.path + args.file_name + timeStamp
    args.save_path = args.save_path + '_val' if not args.train else args.save_path
    return tr_loader, len(train_dataset), ts_loader, len(test_dataset)


def ricker_wavelet(record_time=2000.0, sampling_rate=4.0, f0=0.025):
    nt = int(record_time//sampling_rate) + 1
    t = np.linspace(0, record_time, nt)
    r = (np.pi * f0 * (t - 1 / f0))
    q = np.zeros((nt, 1))
    q[:, 0] = (1 - 2 * r**2) * np.exp(-r**2)
    return q

def read_wavelet(args):
    wavelet = ricker_wavelet(2000.0, 4.0, 0.025)
    # wavelet = np.load('//wsl.localhost/Ubuntu-20.04/home/guanp/wavelet.npy')
    W = scipy.linalg.convolution_matrix(wavelet.squeeze(), 341)[25:341 + 25]
    return wavelet, torch.Tensor(W).to(args.device)
