import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='Path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='Test dataset folder')
parser.add_argument("--test_noiseL", type=float, default=25, help='Noise level for test set')
opt = parser.parse_args()

def normalize(data):
    return data / 255.0

def batch_SSIM(img1, img2, data_range=1.0):
    # img1, img2: torch tensors oblika (1, 3, H, W)
    img1_np = img1.squeeze(0).cpu().numpy()
    img2_np = img2.squeeze(0).cpu().numpy()
    ssim_total = 0
    for c in range(3):
        ssim_total += ssim(img1_np[c], img2_np[c], data_range=data_range)
    return ssim_total / 3

def main():
    print('Loading model ...\n')
    net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    model = nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()

    print('Loading data info ...\n')
    supported_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    files_source = []
    for ext in supported_exts:
        files_source.extend(glob.glob(os.path.join('data', opt.test_data, ext)))
    files_source.sort()

    print(f"Pronađeno {len(files_source)} slika.\n")

    psnr_test = 0
    ssim_test = 0
    valid_count = 0

    for f in files_source:
        print(f"Processing {f}...")

        Img = cv2.imread(f)
        if Img is None or Img.ndim < 3 or Img.shape[2] != 3:
            print(f"Preskačem {f}: nije ispravna RGB slika.\n")
            continue

        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = np.float32(normalize(Img))
        Img = np.transpose(Img, (2, 0, 1))  # (H, W, C) → (C, H, W)

        ISource = torch.Tensor(Img).unsqueeze(0)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.0)
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        with torch.no_grad():
            Out = torch.clamp(INoisy - model(INoisy), 0.0, 1.0)

        psnr = batch_PSNR(Out, ISource, 1.0)
        ssim_val = batch_SSIM(Out, ISource, data_range=1.0)
        psnr_test += psnr
        ssim_test += ssim_val
        valid_count += 1

        print(f"{f} - PSNR: {psnr:.2f} dB, SSIM: {ssim_val:.4f}\n")

    if valid_count > 0:
        psnr_test /= valid_count
        ssim_test /= valid_count
        print(f"\nProsječni PSNR: {psnr_test:.2f} dB")
        print(f"Prosječni SSIM: {ssim_test:.4f}")
    else:
        print("Nijedna ispravna RGB slika nije pronađena za testiranje.")

if __name__ == "__main__":
    main()
