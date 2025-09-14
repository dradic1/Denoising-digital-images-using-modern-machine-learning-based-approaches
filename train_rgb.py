import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *

# Set CUDA environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Argument parser
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='Run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help="Path of log files")
parser.add_argument("--mode", type=str, default="B", help="Known noise level (S) or blind training (B)")
parser.add_argument("--noiseL", type=float, default=25, help="Noise level (ignored when mode=B)")
parser.add_argument("--val_noiseL", type=float, default=25, help="Noise level used on validation set")
opt = parser.parse_args()

def main():
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(reduction='sum')  # Updated for newer PyTorch versions

    model = nn.DataParallel(net).cuda()
    criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    writer = SummaryWriter(opt.outf)

    step = 0
    noiseL_B = [0, 55]  # Used for blind training mode
    best_psnr = 0

    for epoch in range(opt.epochs):
        current_lr = opt.lr if epoch < opt.milestone else opt.lr / 10.
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('Learning rate: %f' % current_lr)

        # Training phase
        for i, data in enumerate(loader_train, 0):
            model.train()
            optimizer.zero_grad()

            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
            else:
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size(0))
                for n in range(noise.size(0)):
                    noise[n] = torch.FloatTensor(noise[0].size()).normal_(mean=0, std=stdN[n] / 255.)

            imgn_train = img_train + noise
            img_train = Variable(img_train.cuda())
            imgn_train = Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())

            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size(0) * 2)
            loss.backward()
            optimizer.step()

            model.eval()
            out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)

            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))

            if step % 10 == 0:
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        # Validation phase
        model.eval()
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
            with torch.no_grad():
                img_val = img_val.cuda()
                noise = noise.cuda()
                imgn_val = img_val + noise
                out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)

        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)

        # Log example images
        out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)

        # Save latest model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

        # Save best model
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net_best.pth'))
            print("Saved best model with PSNR: %.4f" % best_psnr)

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        elif opt.mode == 'B':
            prepare_data(data_path='data', patch_size=60, stride=30, aug_times=1)
    main()
