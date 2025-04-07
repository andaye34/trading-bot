# train.py - Handles data preparation, training, and saving the model

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from model import LSTM, save_model

# Load and preprocess data
ticker = "TSLA"
data = yf.download(ticker, start="2020-09-20", end="2024-03-19")[['Close']]

#ensure date is kept as index
data.index = pd.to_datetime(data.index)

#function to create lagged features
def prepare_dataframe(df, n_steps):
    df = df.copy()
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

#set lookback
lookback = 20
shifted_df = prepare_dataframe(data, lookback)

#normalize the data with MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_np = scaler.fit_transform(shifted_df)

#prepare input(x) and target(y)
x = shifted_df_np[:, 1:] #input features(previous close prices)
y = shifted_df_np[:, 0] #Target(the next close price we are trying to predict
#reshape correctly
x = np.flip(x, axis=1) # Flip sequence for chronological order
split_index = int(len(x) * 0.80) #split into 80% training and 20 for testing
X_train, X_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

#reshape to fit LSTM input format:(batch, sequence_length, input_dim)
X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Convert to torch tensors
X_train, X_test = torch.tensor(X_train.copy()).float(), torch.tensor(X_test.copy()).float()
y_train, y_test = torch.tensor(np.array(y_train)).float(), torch.tensor(np.array(y_test)).float()

#test to see the shape
print(f"X_Train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
#another test to ensure training data matches target data
print(f"Training samples: {len(X_train)}, Test Sample: {len(X_test)}")
print(f"Training Labels: {len(y_train)}, Test Labels: {len(y_test)}")
#dataset class for Pytorch Dataloader
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

#create training and testing data sets
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

#set up model, loss function and optimizer
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = LSTM(1, 8, 1).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#training function
def train_one_epoch():
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Training Loss: {running_loss / len(train_loader):.6f}')

#validation function
def validate_one_epoch():
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            running_loss += loss.item()
    print(f'Validation Loss: {running_loss / len(test_loader):.6f}')
#the training loop
for epoch in range(100):
    print(f'Epoch {epoch + 1}/100')
    train_one_epoch()
    validate_one_epoch()

# Save trained model
save_model(model)

