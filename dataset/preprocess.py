import torch
import torch.nn as nn
import torch.nn.functional as F


N_TARGET_FRAMES = 124
N_COLUMNS = 1629
class PreprocessLayer(nn.Module):
    def __init__(self):
        super(PreprocessLayer, self).__init__()

    def forward(self, data0, resize=True):
        # Fill NaN Values With 0
        data0 = torch.tensor(data0)
        data = torch.where(torch.isnan(data0), torch.tensor(0.0), data0)

        # Add another dimension

        # # Empty Hand Frame Filtering
        # hands = data[:, :, :, :84].abs()
        # mask = hands.sum(dim=2) != 0
        # data = data[mask].unsqueeze(0)

        # Padding with Zeros
        N_FRAMES = data.shape[0] 
        if N_FRAMES < N_TARGET_FRAMES:
            zeros_tensor = torch.zeros(N_TARGET_FRAMES - N_FRAMES, N_COLUMNS, dtype=torch.float32)
            data = torch.cat((data, zeros_tensor), dim=0)
        data = data[None]
        tensor_downsampled = F.interpolate(data.unsqueeze(0), size=(N_TARGET_FRAMES, N_COLUMNS), mode='bilinear', align_corners=False)[0]
        data = tensor_downsampled.squeeze(axis=0)
        return data
