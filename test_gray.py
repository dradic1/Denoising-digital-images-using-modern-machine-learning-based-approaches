import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from skimage.metrics import structural_similarity as compare_ssim
from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='Test dataset name')
parser.add_argument("--test_noiseL", type=float, default=25, help='Noise level for test set')
opt = parser.parse_args()

def normalize(data):
    return data / 255.0

def load_images(folder):
    supported_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    files = []
    for ext in supported_exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files.sort()
    return files

def main():
    print('🔧 Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    model = nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()

    print('🖼️ Loading test images ...\n')
    test_folder = os.path.join('data', opt.test_data)
    files_source = load_images(test_folder)
    print(f"Found {len(files_source)} images.\n")

    psnr_total = 0
    ssim_total = 0
    valid_count = 0

    for f in files_source:
        Img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if Img is None:
            print(f"⚠️ Skipping {f}: invalid image.")
            continue

        Img = normalize(np.float32(Img))
        Img = np.expand_dims(Img, 0)  # (H, W) → (1, H, W)
        Img = np.expand_dims(Img, 0)  # (1, H, W) → (1, 1, H, W)
        ISource = torch.Tensor(Img)

        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        INoisy = ISource + noise

        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        with torch.no_grad():
            Out = torch.clamp(INoisy - model(INoisy), 0.0, 1.0)

        psnr = batch_PSNR(Out, ISource, 1.0)

        out_np = Out.cpu().numpy().squeeze()
        target_np = ISource.cpu().numpy().squeeze()
        ssim_val = compare_ssim(out_np, target_np, data_range=1.0)

        psnr_total += psnr
        ssim_total += ssim_val
        valid_count += 1

        print(f"{os.path.basename(f)} - PSNR: {psnr:.2f} dB | SSIM: {ssim_val:.4f}")

    if valid_count > 0:
        avg_psnr = psnr_total / valid_count
        avg_ssim = ssim_total / valid_count
        print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
    else:
        print("No valid images processed.")

if __name__ == "__main__":
    main()
