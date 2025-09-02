####################################################################################
# 라이브러리 세팅
####################################################################################

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import random
import struct
from glob import glob
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from torchvision.transforms import functional as TF
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
target_size = (128, 128)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

####################################################################################
# dat-plot.py 파일 활용
####################################################################################

def load_to_file(file_name):
    with open(file_name, 'rb') as fin:
        width = int.from_bytes(fin.read(4), byteorder='little')
        height = int.from_bytes(fin.read(4), byteorder='little')

        data_size = width * height

        pData = [struct.unpack('f', fin.read(4)) for _ in range(data_size)]
        pData = np.array(pData).reshape((height, width, 1))

        pData = pData.astype(np.float32)

        return pData

dat_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/2D_1309AI06__MNT_2177__3D_MergeDepth.dat')
east_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/3D_Depth_E.dat')
west_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/3D_Depth_N.dat')
south_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/3D_Depth_S.dat')
north_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/3D_Depth_W.dat')

dat_dir = sorted(dat_dir)
east_dir = sorted(east_dir)
west_dir = sorted(west_dir)
south_dir = sorted(south_dir)
north_dir = sorted(north_dir)

train_set = []

for y, e, w, s, n in tqdm(zip(dat_dir, east_dir, west_dir, south_dir, north_dir), total=len(dat_dir)):
    Y = load_to_file(y)
    E = load_to_file(e)
    W = load_to_file(w)
    S = load_to_file(s)
    N = load_to_file(n)

####################################################################################
# 이미지를 DAT 형식으로 저장하기
# -> 이를 통해 이미지 to DAT / DAT to 이미지 변환이 가능
####################################################################################

def save_to_dat_file(file_name, data):
    data = np.array(data)
    data = np.mean(np.array(data), axis=2)
    height, width = data.shape

    data = data.astype(np.float32)

    with open(file_name, 'wb') as fout:
        fout.write(width.to_bytes(4, byteorder='little'))
        fout.write(height.to_bytes(4, byteorder='little'))

        for row in data:
            for pixel in row:
                fout.write(struct.pack('f', pixel.item()))

save_to_dat_file('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/test-dat.dat', Y)
DAT = load_to_file('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/test-dat.dat')
#plt.imshow(DAT)
#plt.show()


