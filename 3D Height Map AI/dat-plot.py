# 라이브러리 로드
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import struct
from glob import glob
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from torchvision.transforms import functional as TF

# DAT 파일 로드 및 시각화 (파일 경로 입력)
def load_to_file(file_name):
    with open(file_name, 'rb') as fin:
        width = int.from_bytes(fin.read(4), byteorder='little')
        height = int.from_bytes(fin.read(4), byteorder='little')

        data_size = width * height

        pData = [struct.unpack('f', fin.read(4)) for _ in range(data_size)]
        pData = np.array(pData).reshape((height, width, 1))

        pData = pData.astype(np.float32)

        plt.imshow(pData)
        plt.show()

        return pData

# DAT 파일 전부 불러오기
dat_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/2D_1309AI06__MNT_2177__3D_MergeDepth.dat')
east_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/3D_Depth_E.dat')
west_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/3D_Depth_N.dat')
south_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/3D_Depth_S.dat')
north_dir = glob('C:/Users/TJKim/3D HeightMap AI (인수인계)/데이터/3D_Depth_W.dat')

# DAT 파일 정렬
dat_dir = sorted(dat_dir)
east_dir = sorted(east_dir)
west_dir = sorted(west_dir)
south_dir = sorted(south_dir)
north_dir = sorted(north_dir)

# DAT 파일 이미지로 시각화
for y, e, w, s, n in tqdm(zip(dat_dir, east_dir, west_dir, south_dir, north_dir), total=len(dat_dir)):
    Y = load_to_file(y)
    E = load_to_file(e)
    W = load_to_file(w)
    S = load_to_file(s)
    N = load_to_file(n)
