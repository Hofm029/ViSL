import torch
import torch.nn as nn
import math

class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,padding=kernel_size//2, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, in_channels, 1)
        self.bn = nn.BatchNorm1d(in_channels)  # Batch Normalization layer

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)  # Applying Batch Normalization
        return out

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=k, padding=k // 2, groups=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class DepthwiseConv1DECA(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv1DECA, self).__init__()
        self.depthwise_conv = DepthwiseConv1D(in_channels, kernel_size)
        self.eca = ECABlock(in_channels)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.eca(out)
        return out

class DepthwiseConv1DModel(nn.Module):
    def __init__(self, in_channels, kernel_size,dropout_prob=0.1):
        #embedding
        super(DepthwiseConv1DModel, self).__init__()
        self.embedding = nn.Linear(in_channels, in_channels)
        self.conv1 = DepthwiseConv1DECA(in_channels=in_channels, kernel_size=kernel_size) 
        self.dense = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout_prob)
        # droput
        #skip
        self.in_channels = in_channels

    def forward(self, x):
        skip = x
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.dense(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        x = x + skip
        return x
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, in_channels,expand=4,dropout=0.2):
        super(TransformerBlock, self).__init__()
        self.dense = nn.Linear(in_channels, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*expand
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )
        self.bnorm = nn.BatchNorm1d(d_model)
        self.lnorm = nn.LayerNorm(d_model)  # Add LayerNorm
        self.dropout = nn.Dropout(dropout)  # Add Dropout
        self.fc1 = nn.Linear(d_model, d_model*expand)  # Add Linear layer for skip connection
        self.fc2 = nn.Linear(d_model*expand, in_channels)  # Add Linear layer for skip connection

    def forward(self, x):
        x = self.dense(x.permute(0, 2, 1))
        skip1 = x
        x = self.transformer_encoder(self.dropout(self.lnorm(x))) + skip1
        x = self.dropout(self.fc2(self.fc1(x)))
        return x.permute(0, 2, 1)
class LateDropout(nn.Module):
    def __init__(self, p=0.5, start_step=0):
        super(LateDropout, self).__init__()
        self.p = p
        self.start_step = start_step

    def forward(self, x):
        if not self.training or torch.randint(0, x.size(0), (1,)).item() >= self.start_step:
            return x
        mask = torch.rand(x.size(0), 1, x.size(2), device=x.device) < self.p
        return x.masked_fill_(mask, 0) / (1 - self.p)
# Tiếp tục triển khai các lớp còn lại của mô hình
class Net(nn.Module):
    def __init__(self,in_channels,d_model, kernel_size,num_layers,num_heads,num_classes,dropout_step=0):
        super(Net, self).__init__()
        self.depcov1 = DepthwiseConv1DModel(in_channels, kernel_size)
        self.depcov2 = DepthwiseConv1DModel(in_channels, kernel_size)
        self.depcov3 = DepthwiseConv1DModel(in_channels, kernel_size)
        self.tfm1    = TransformerBlock(d_model, num_layers, num_heads,in_channels, expand=4)
        self.depcov4 = DepthwiseConv1DModel(in_channels, kernel_size)
        self.depcov5 = DepthwiseConv1DModel(in_channels, kernel_size)
        self.depcov6 = DepthwiseConv1DModel(in_channels, kernel_size)
        self.tfm2    = TransformerBlock(d_model, num_layers, num_heads,in_channels, expand=4)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # GlobalAveragePooling1D
        self.late_dropout = LateDropout(p=0.8, start_step=dropout_step)
        self.classifier = nn.Linear(in_channels, num_classes, bias=True)
    def forward(self, x):
      x = x.permute(0,2,1)
      x = self.depcov1(x)
      x = self.depcov2(x)
      x = self.depcov3(x)
      x = self.tfm1(x)
      x = self.depcov4(x)
      x = self.depcov5(x)
      x = self.depcov6(x)
      x = self.tfm2(x)
      x = self.global_avg_pool(x).squeeze(2)
      x = self.late_dropout(x)
      x = self.classifier(x)
      return x
# Test the model with random input
# conv_block  = DepthwiseConv1DModel(in_channels=390, kernel_size=17)
if __name__ == "__main__":
    conv_block = Net( in_channels=390,
                    d_model=256,
                    kernel_size=17,
                    num_layers=1,
                    num_heads=4,
                    num_classes=25)
    input_data = torch.randn(1, 124, 390)  # Example input size
    output = conv_block(input_data)
    print("Output shape:", output.shape)
    # Count the total number of parameters
    total_params = sum(p.numel() for p in conv_block.parameters())
    print(f"Total parameters: {total_params}")