# model.py - Defines the LSTM model architecture
import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()  # Fixed indentation here

    def forward(self, x):
        batch_size = x.shape[0]
        # Initialize hidden state(h0) and cell state (c0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Extract last time step's output
        out = out[:, -1, :]  # Taking only the last timestep output
        # Pass through fully connected layer
        out = self.fc(out)
        # Apply tanh activation to keep in range of -1 and 1
        out = torch.tanh(out)
        return out

# Save model function
def save_model(model, path="saved_models/lstm_model.pth"):
    torch.save(model.state_dict(), path)

# Load model function
def load_model(path="saved_models/lstm_model.pth"):
    model = LSTM(1, 8, 1)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model
