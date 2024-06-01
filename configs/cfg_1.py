import os
import sys
from importlib import import_module
import platform
import json
import numpy as np
import torch
import pandas as pd
import augmentations as A
from types import SimpleNamespace
cfg = SimpleNamespace(**{})
cfg.debug = True

#paths
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.output_dir = f"output/weights/{os.path.basename(__file__).split('.')[0]}"
cfg.data_folder = f"dataset/train_landmarks_npy/"
cfg.train_df = f'dataset/train_folded.csv'
cfg.symmetry_fp = 'dataset/symmetry.csv'

# stages
cfg.test = False
cfg.train = True
cfg.train_val =  False
cfg.eval_epochs = 1
cfg.seed = -1

#logging
cfg.neptune_project = "common/quickstarts"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"

# DATASET
cfg.dataset = "ds_1"
cfg.min_seq_len = 15

cfg.max_len = 124
cfg.max_phrase = 31 + 2 #max of train data + SOS + EOS

#model

cfg.input_size = 390
cfg.hidden_size = 1024
cfg.output_size = 10


cfg.model = "model_1"
cfg.ce_ignore_index = -100
cfg.label_smoothing = 0.
cfg.n_landmarks = 130
cfg.return_logits = False
cfg.pretrained = True
cfg.val_mode = 'padded'


#Encoder 
cfg.n_target_frames = 124
cfg.n_columns = 1086

cfg.means = 0.5
cfg.stds = 0.5
cfg.units_encoder = 512
cfg.nhead = 4
cfg.num_layers = 2
cfg.dropout = 0.2
cfg.classifier_class = 15

#transformer model
cfg.d_model = 256  
cfg.n_landmark = 390
cfg.dim_feedforward = 512
cfg.num_layers = 2
cfg.num_heads = 2
cfg.num_classes = 10

# OPTIMIZATION & SCHEDULE
cfg.fold = 0
cfg.epochs =100
cfg.lr = 5e-4 * 9
cfg.lr_max = 0.01
cfg.num_cycles = 0.5
cfg.warmup_status = True
cfg.optimizer = "AdamW"
cfg.weight_decay = 0.05
cfg.clip_grad = 4.
cfg.nwarmup = 10
cfg.batch_size = 32
cfg.batch_size_val = 128
cfg.mixed_precision = True # True
cfg.pin_memory = False
cfg.grad_accumulation = 8.
cfg.num_workers = 8
cfg.track_grad_norm = False #True
#Saving
cfg.save_weights_only = True
cfg.save_only_last_ckpt = True

cfg.decoder_mask_aug = 0.2
cfg.flip_aug = 0.5
cfg.outer_cutmix_aug = 0.5


cfg.val_aug = None


cfg.train_aug = A.Compose([A.Resample(sample_rate=(0.5,1.5), p=0.8),
                           A.SpatialAffine(scale=(0.8,1.2),shear=(-0.15,0.15),shift=(-0.1,0.1),degree=(-30,30),p=0.75),  
                           A.TemporalMask(size=(0.2,0.4),mask_value=0.,p=0.5), #mask with 0 as it is post-normalization
                           A.SpatialMask(size=(0.05,0.1),mask_value=0.,mode='relative',p=0.5), #mask with 0 as it is post-normalization
                          ])
cfg.train_aug._disable_check_args() #disable otherwise input must be numpy/ int8
