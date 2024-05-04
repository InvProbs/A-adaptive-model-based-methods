from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
import random, torch
import numpy as np


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def center_crop(data, shape):
    assert 0 < shape[0] <= data.shape[0]
    assert 0 < shape[1] <= data.shape[1]
    w_from = (data.shape[0] - shape[0]) // 2
    h_from = (data.shape[1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[w_from:w_to, h_from:h_to, ...]


def load_data(args):
    timeStamp = datetime.now().strftime("%Y%m%d-%H%M")
    args.path = args.path + args.file_name

    if args.dataset == "CelebA":
        # Downloaded from https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
        save_path = args.path + 'CelbeA_' + timeStamp
        save_path += '_ts' if not args.train else ''
        data_path = 'E:/data/CelebA/'#img_align_celeba'

        transform = transforms.Compose([transforms.Resize((120, 100)), transforms.CenterCrop((110, 88)), transforms.ToTensor()])
        CelebA_dataset = datasets.ImageFolder(data_path, transform=transform)  # image size: 218 x 178
        full_length = len(CelebA_dataset)
        train_index = range(int(full_length * 0.9))
        val_index = range(int(full_length * 0.8), int(full_length * 0.9))
        # test_index = range(int(full_length * 0.9), full_length)
        test_index = range(32*20)

        train_set = torch.utils.data.Subset(CelebA_dataset, train_index)
        val_set = torch.utils.data.Subset(CelebA_dataset, val_index)
        test_set = torch.utils.data.Subset(CelebA_dataset, test_index)

        tr_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size_val, shuffle=False, drop_last=True)
        ts_loader = DataLoader(dataset=test_set, batch_size=args.batch_size_val, shuffle=False, drop_last=True)

        return tr_loader, len(train_set), val_loader, len(val_set), ts_loader, len(test_set), save_path
