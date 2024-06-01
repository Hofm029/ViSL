import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
class Preprocessing(nn.Module):
    def __init__(self):
        super(Preprocessing, self).__init__()

    def normalize(self,x):
        nonan = x[~torch.isnan(x)].view(-1, x.shape[-1])
        x = x - nonan.mean(0)[None, None, :]
        x = x / nonan.std(0, unbiased=False)[None, None, :]
        return x
    
    def fill_nans(self,x):
        x[torch.isnan(x)] = 0
        return x
        
    def forward(self, x):
        
        #seq_len, 3* n_landmarks -> seq_len, n_landmarks, 3
        x = x.reshape(x.shape[0],3,-1).permute(0,2,1)

        # Normalize & fill nans
        x = self.normalize(x)
        x = self.fill_nans(x)
        return x
def interpolate_or_pad(data, max_len=100, mode="start"):
    diff = max_len - data.shape[0]

    if diff <= 0:  # Crop
        data = F.interpolate(data.permute(1,2,0),max_len).permute(2,0,1)
        mask = torch.ones_like(data[:,0,0])
        return data, mask
    
    coef = 0
    padding = torch.ones((diff, data.shape[1], data.shape[2]))
    mask = torch.ones_like(data[:,0,0])
    data = torch.cat([data, padding * coef])
    mask = torch.cat([mask, padding[:,0,0] * coef])
    return data, mask


def flip(data, flip_array):
    
    data[:,:,0] = -data[:,:,0]
    data = data[:,flip_array]
    return data

def draw_data(data):
    # Vẽ biểu đồ
    print(data.shape)
    plt.imshow(data, interpolation='nearest', origin='upper')
    plt.colorbar()
    plt.show()
def draw_data_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.where(data > 0)
    intensity = data[data > 0]  # Lấy giá trị cường độ từ dữ liệu
    
    # Tính toán màu sắc dựa trên cường độ
    norm = plt.Normalize(intensity.min(), intensity.max())
    cmap = plt.cm.viridis  # Chọn colormap, bạn có thể thay đổi nếu cần
    
    # Vẽ dữ liệu 3D với màu sắc dựa trên cường độ
    ax.scatter(x, y, z, c=intensity, cmap=cmap, norm=norm)
    
    # Đặt nhãn cho các trục
    ax.set_xlabel('n_frames')
    ax.set_ylabel('n_landmark')
    ax.set_zlabel('channels')
    
    # Tạo thanh màu (colorbar)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Intensity')

    plt.show()
def visualize_data(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    channels = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].imshow(data[:,:,i])  # hiển thị từng kênh
        axes[i].set_title(f'Channel {channels[i]}')  # đặt tiêu đề cho từng kênh
        axes[i].set_xlabel('n_frame')  # đặt tên cho trục x
        axes[i].set_ylabel('n_landmark')  # đặt tên cho trục y
    plt.show()
    

class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug=None, mode="train"):

        self.cfg = cfg
        self.df = df.copy()
        self.mode = mode
        self.aug = aug
        
        #input stuff
        with open(cfg.data_folder + 'inference_args.json', "r") as f:
            columns = json.load(f)['selected_columns']
        
        self.xyz_landmarks = np.array(columns)
        landmarks = np.array([item[2:] for item in self.xyz_landmarks[:len(self.xyz_landmarks)//3]])
        
        symmetry = pd.read_csv(cfg.symmetry_fp).set_index('id')
        flipped_landmarks = symmetry.loc[landmarks]['corresponding_id'].values
        self.flip_array = np.where(landmarks[:,None]==flipped_landmarks[None,:])[1]
        self.max_len = cfg.max_len
        self.processor = Preprocessing()
        # self.preprocesslayer = PreprocessLayer(N_TARGET_FRAMES, N_COLS0 )
        
        #target stuff
        self.flip_aug = cfg.flip_aug

        if mode == "test":
            self.data_folder = cfg.test_data_folder
        else:
            self.data_folder = cfg.data_folder
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        file_id, sequence_id,label = row[['file_id','sequence_id','label']]
        data = self.load_one(file_id, sequence_id)
        data = torch.from_numpy(data)
        print("DATA RAW")
        draw_data(data)   ## Check
        
        if self.mode == 'train':
            data = self.processor(data)
            print("After Preprocess")
            
            draw_data(data)   ## Check
            # draw_data_3d(data)   ## Check
            visualize_data(data)
            if np.random.rand() < self.flip_aug:
                data = flip(data, self.flip_array) 
                print("After Flip")
                # draw_data_3d(data)   ## Check
                visualize_data(data)

            if self.aug:
                data = self.augment(data)
                print("After augment")
                # draw_data_3d(data)   ## Check
                visualize_data(data)
                
        else: 
            data = self.processor(data)
        
        data, mask = interpolate_or_pad(data, max_len=self.max_len)
        # data =  self.preprocesslayer(data.reshape(data.shape[0],-1))
        feature_dict = {'input':data,
                        'output': label,
                        }
        return data.float() , label
    
    def augment(self,x):
        x_aug = self.aug(image=x.float())['image']
        return x_aug
    
    def load_one(self, file_id, sequence_id):
        path = self.data_folder + f'{file_id}/{sequence_id}.npy'
        data = np.load(path) # seq_len, 3* nlandmarks
        return data

if __name__ == '__main__':
    pass