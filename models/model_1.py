import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Lấy output của LSTM ở thời điểm cuối cùng
        return output
    
if __name__ == "__main__":
    input_size = 390  # Số lượng đặc trưng
    hidden_size = 124
    output_size = 9    
    model = SimpleLSTM(input_size, hidden_size, output_size)
    x = torch.rand((1,130,390))
    print(model(x).shape)