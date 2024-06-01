import torch
import torch.nn as nn
import torch.nn.functional as F
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
        x = torch.Tensor(x)
        #seq_len, 3* n_landmarks -> seq_len, n_landmarks, 3
        x = x.reshape(x.shape[0],3,-1).permute(0,2,1)
        
        # Normalize & fill nans
        x = self.normalize(x)
        x = self.fill_nans(x)
        return x
def interpolate_or_pad(data, max_len=124, mode="start"):
    diff = max_len - data.shape[0]

    if diff <= 0:  # Crop
        data = F.interpolate(data.permute(1,2,0),max_len).permute(2,0,1)
        return data
    
    coef = 0
    padding = torch.ones((diff, data.shape[1], data.shape[2]))
    data = torch.cat([data, padding * coef])
    return data