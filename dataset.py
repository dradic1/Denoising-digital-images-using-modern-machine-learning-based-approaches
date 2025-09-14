import os
import glob
import numpy as np
import random
import h5py
import torch
import cv2
import torch.utils.data as udata
from utils import data_augmentation

def normalize(data):
    return data / 255.0

def Im2Patch(img, win, stride=1):
    k = 0
    endc, endw, endh = img.shape
    patch = img[:, 0:endw-win+1:stride, 0:endh-win+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch).reshape(endc, TotalPatNum)
            k += 1
    
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    print('Processing training data...')
    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'train', '*.jpg'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0

    for file in files:
        img = cv2.imread(file)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape  

        for scale in scales:
            Img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            Img = np.transpose(Img, (2, 0, 1))
            Img = np.float32(normalize(Img))  
            patches = Im2Patch(Img, win=patch_size, stride=stride)

            print(f"File: {file}, Scale: {scale:.1f}, # Patches: {patches.shape[3] * aug_times}")

            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1

                for m in range(aug_times - 1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(f"{train_num}_aug_{m+1}", data=data_aug)
                    train_num += 1

    h5f.close()

    print('\nProcessing validation data...')
    files = glob.glob(os.path.join(data_path, 'val', '*.jpg'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0

    for file in files:
        print(f"File: {file}")
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = np.transpose(img, (2, 0, 1))  
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1

    h5f.close()
    print(f"Training set: {train_num} samples")
    print(f"Validation set: {val_num} samples\n")

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        self.h5_file = 'train.h5' if train else 'val.h5'
        self.h5f = h5py.File(self.h5_file, 'r')
        self.keys = list(self.h5f.keys())
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        data = np.array(self.h5f[key])
        return torch.Tensor(data)

    def close(self):
        self.h5f.close()
