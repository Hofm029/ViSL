import os
import glob
import gc
from copy import copy
import numpy as np
import pandas as pd
import importlib
import sys
from tqdm import tqdm, notebook
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader,TensorDataset
from collections import defaultdict
import transformers
# from decouple import Config, RepositoryEnv
import random
from utils import *
import torchvision.models as models


BASEDIR= './'#'../input/asl-fingerspelling-config'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')


parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename", default="cfg_3")
parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)

CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
Net = importlib.import_module(cfg.model).Net



cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(cfg.device)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection  import train_test_split
train_df = pd.read_csv('./dataset/train.csv')
val_df = pd.read_csv('./dataset/test.csv')
# Sử dụng LabelEncoder để chuyển đổi cột 'phrase' thành số
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['phrase'])
val_df['label'] = label_encoder.fit_transform(val_df['phrase'])
# Tách dữ liệu ra thành tập huấn luyện và tập kiểm thử

# Set up the dataset and dataloader
train_dataset = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
val_dataset = CustomDataset(val_df, cfg, aug=cfg.train_aug, mode="val")

train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory= True
    )
val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory= True
    )



model = Net(num_block=cfg.num_block,
        num_class=cfg.num_class,
        num_landmark=cfg.num_landmark,
        max_length=cfg.max_length,
        embed_dim=cfg.embed_dim,
        num_head=cfg.num_head, 
        in_channels=cfg.in_channels, 
        kernel_size=cfg.kernel_size).to(cfg.device)

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_max, weight_decay=cfg.weight_decay)
criterion = torch.nn.CrossEntropyLoss()
scaler = GradScaler()


# Start the training and validation loop
cfg.curr_step = 0
optimizer.zero_grad()
total_grad_norm = None    
total_grad_norm_after_clip = None
i = 0 

index = 1
while os.path.exists(new_path := f"{cfg.path}_{index}/"):
    index += 1
os.makedirs(new_path)
path = new_path
print("All save in: ",path)

LR_SCHEDULE = [lrfn(step, num_warmup_steps=cfg.nwarmup, lr_max=cfg.lr_max,num_training_steps=cfg.epochs, num_cycles=cfg.num_cycles) for step in range(cfg.epochs)]
plot_lr_schedule(LR_SCHEDULE, cfg.epochs)
iters  = []
losses = []
topk_accuracy = []
#   val_acc_list = []q
learning_rates=[]
n = 0

for epoch in range(cfg.epochs):
    if cfg.warmup_status == True:
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR_SCHEDULE[epoch]
    learning_rates.append(optimizer.param_groups[0]['lr'])
    for inputs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{cfg.epochs}', unit='batch'):
        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
        model.train()
        out = model(inputs)

        loss = criterion(out, labels.long()) # compute the total loss
        loss.backward()
        optimizer.step()              # make the updates for each parameter
        optimizer.zero_grad()         # a clean up step for PyTorch
        iters.append(n)
        losses.append(float(loss)/cfg.batch_size) # compute *average* loss

        with torch.no_grad():
            total = 0
            corrects = {k: 0 for k in cfg.topk}
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
                outputs = model(inputs)
                _, predicted = torch.cfg.topk(outputs, max(int(x) for x in cfg.topk), dim=1)
                total += labels.size(0)
                for k in cfg.topk:
                    corrects[k] += torch.sum(torch.any(predicted[:, :int(k)] == labels.view(-1, 1), dim=1)).item()
            accuracies = [(corrects[k] / total)  for k in cfg.topk]
        topk_accuracy.append(accuracies)
        if topk_accuracy[n][0] > best_accuracy :
          best_accuracy = topk_accuracy[n][0]
          best_model = model
          torch.save({"model": best_model.state_dict()},path+f"checkpoint_best_seed{cfg.seed}.pth")
        n += 1
    print(" ".join([f"top{k}_accuracy: {round(topk_accuracy[n-1][i],4)} -  " for i, k in enumerate(topk)])+f'train_loss: {round(losses[-1],3)} -  learning_rate: {round(learning_rates[-1],7)}\n ')
    print("The best accuracy is: ", best_accuracy)

draw_plot(cfg.epochs,learning_rates,cfg.batch_size,iters,topk, topk_accuracy,losses,path)
torch.save({"model": model.state_dict()},path+f"checkpoint_last_seed{cfg.seed}.pth")

input_data = torch.randn(1,124, 390)
model(torch.randn(1,124, 390))
# Chuyển model sang ONNX
torch.onnx.export(model,                      # model
                    input_data,                # dữ liệu đầu vào mẫu
                    path+"model.onnx",              # tên file output
                    export_params=True,        # chuyển cả trọng số của model
                    do_constant_folding=True,  # folding các biến hằng trong model để tối ưu
                    input_names=['input'],     # tên của các đầu vào của model
                    output_names=['output'])   # tên của các đầu ra của model
print(f"Checkpoint save : " +  path+f"checkpoint_last_seed{cfg.seed}.pth")