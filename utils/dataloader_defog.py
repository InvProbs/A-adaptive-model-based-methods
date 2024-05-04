import numpy as np
import cv2, glob, torch
import matplotlib, os, datetime
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, foggy_dir, depth_dir, gt_dir, transform=None):
        self.foggy_dir = foggy_dir
        self.gt_dir = gt_dir
        self.depth_dir = depth_dir
        self.transform = transform

        foggy_file_list = glob.glob(foggy_dir + "*")
        self.foggy_data = []
        for class_path in foggy_file_list:
            for img_path in glob.glob(class_path + "/*.png"):
                self.foggy_data.append(img_path)

        gt_file_list = glob.glob(gt_dir + "*")
        self.gt_data = []
        for class_path in gt_file_list:
            for img_path in glob.glob(class_path + "/*.png"):
                self.gt_data.append(img_path)

        depth_file_list = glob.glob(depth_dir + "*")
        self.depth_data = []
        for class_path in depth_file_list:
            for img_path in glob.glob(class_path + "/*.png"):
                self.depth_data.append(img_path)

    def __len__(self):
        return len(self.foggy_data)

    def __getitem__(self, idx):
        # path = os.path.join(self.root_dir, self.files[idx])
        foggy_path, depth_path, gt_path = self.foggy_data[idx], self.depth_data[idx//3], self.gt_data[idx//3]
        foggy_img = self.transform(Image.open(foggy_path))
        depth_img = self.transform(Image.open(depth_path))
        gt_img = self.transform(Image.open(gt_path))
        return foggy_img, depth_img, gt_img


def load_foggy_data(args):
    train_foggy_path = '../data/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/train/'
    train_gt_path = '../data/leftImg8bit_trainvaltest/leftImg8bit/train/'
    train_depth_path = '../data/disparity_trainvaltest/disparity/train/'
    val_foggy_path = '../data/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/val/'
    val_gt_path = '../data/leftImg8bit_trainvaltest/leftImg8bit/val/'
    val_depth_path = '../data/disparity_trainvaltest/disparity/val/'
    test_foggy_path = '../data/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/test/'
    test_gt_path = '../data/leftImg8bit_trainvaltest/leftImg8bit/test/'
    test_depth_path = '../data/disparity_trainvaltest/disparity/test/'
    # transform = transforms.Compose([transforms.Resize([256, 512]), transforms.ToTensor()])
    transform = transforms.Compose([transforms.Resize([128, 256]), transforms.ToTensor()])
    train_dataset = CustomDataset(train_foggy_path, train_depth_path, train_gt_path, transform)
    val_dataset = CustomDataset(val_foggy_path, val_depth_path, val_gt_path, transform)
    test_dataset = CustomDataset(test_foggy_path, test_depth_path, test_gt_path, transform)

    tr_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size_val, shuffle=True, drop_last=True)
    ts_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_val, shuffle=False, drop_last=True)

    timeStamp = datetime.datetime.now().strftime("%m%d-%H%M")
    args.save_path = args.path + args.file_name + 'k=' + str(args.maxiters) + '_' + timeStamp
    if not args.train:
        args.save_path += '_val'
    return tr_loader, len(train_dataset), val_loader, len(val_dataset), ts_loader, len(test_dataset)