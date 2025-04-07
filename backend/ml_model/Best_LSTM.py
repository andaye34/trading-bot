# Imports.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# Path to the CSV file
import yfinance as yf

ticker = "GME"


# Load the data into a DataFrame

data = yf.download(ticker, start="2020-01-01", end = "2024-01-01")
data = data[['Close']]


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


plt.figure(figsize=(20, 10))
plt.subplot(2,1,1)

data.reset_index(inplace = True)
data['Date'] =pd.to_datetime(data['Date'])
plt.plot(data['Date'], data['Close'])

from copy import deepcopy as dc
def prepare_dataframe(df, n_steps):
  df = dc(df)
  df.index = pd.to_datetime(df.index) #ensure idex is datetime


  for i in range(1,n_steps+1):
    df[f'Close(t-{i})'] = df['Close'].shift(i)

  df.dropna(inplace= True)
  return df

lookback = 20
shifted_df = prepare_dataframe(data,lookback)
shifted_df_np = shifted_df.drop(columns=["Date"]).to_numpy()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_np = scaler.fit_transform(shifted_df_np)



x = shifted_df_np[:, 1:]
y = shifted_df_np[:, 0]
x = dc(np.flip(x, axis =1))
x.shape, y.shape


split_index = int(len(x) * 0.80)



X_train = x[:split_index]
X_test = x[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train = X_train.reshape((-1, lookback, 1))
X_test = X_test.reshape((-1, lookback, 1))

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))


X_train.shape, X_test.shape, y_train.shape, y_test.shape

#wrap in pytorch tensors
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()

y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

X_train.shape, X_test.shape, y_train.shape, y_test.shape


from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
  def __init__(self, x, y):
    self.x = x #Matrix x
    self.y = y # output vector y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, i):
    return self.x[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train,y_train) #Pass in dataset
test_dataset = TimeSeriesDataset(X_test,y_test)


from torch.utils.data import DataLoader

batch_size = 50
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
  x_batch, y_batch = batch[0] , batch[1].to(device) #put it on the device cpu in this case
  print(x_batch.shape, y_batch.shape)
  break

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super(). __init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, 1)

  def forward(self, x):
    batch_size = x.shape[0]
    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    return out

model = LSTM(1,8,1)
model.to(device)


def train_one_epoch():
  model.train(True)
  print(f'Epoch: {epoch + 1}')
  running_loss = 0.0

  for batch_index, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    output = model(x_batch)
    loss = loss_fn(output, y_batch)
    running_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_index % 100 == 99:
      avg_loss_across_batches = running_loss / 100
      print('Batch {0}, Loss: {1:.3f}'.format(batch_index + 1, avg_loss_across_batches))
      running_loss = 0.0

  print()


def validate_one_epoch():
  model.train(False)
  running_loss = 0.0

  for batch_index, batch in enumerate(test_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

    with torch.no_grad():
      output = model(x_batch)
      loss = loss_fn(output, y_batch)
      running_loss += loss.item()

  avg_loss_across_batches = running_loss / len(test_loader)
  print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
  print('**************************************************')
  print()

lr = 0.001
epochs = 100
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    train_one_epoch()
    validate_one_epoch()


#tester with new future unseen data
future_data = yf.download("GME", start="2024-01-01", end="2024-02-01")[['Close']]
future_data_scaled = scaler.transform(future_data)
future_x = torch.tensor(future_data_scaled).float().to(device)

with torch.nograd():
	future_pred = model(future_x).cpu().numpy()
plt.figure(figsize=(20, 10))
plt.plot(future_data.index, future_data, label="actual Close")
plt.plot(future_data.index, future_preds, label="predicted Close")
plt.xlabel('date')
plt.ylabel('close')
plt.legend()
plt.savefig("unseen_data.png")


with torch.no_grad():
  predicted = model(X_train.to(device)).to('cpu').numpy()
plt.figure(figsize=(20, 10))
plt.plot(y_train, label = 'Actual Close')
plt.plot(predicted, label = 'Predicted Close')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()

plt.savefig("prediction_plot.png")

train_predictions = predicted.flatten()

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)

train_predictions = dc(dummies[:,0])


dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = np.array(y_train).flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dc(dummies[:,0])


plt.figure(figsize=(20, 10))
plt.plot(new_y_train, label = 'Actual Close')
plt.plot(train_predictions, label = 'Predicted Close')
plt.xlabel('Close')
plt.ylabel('Day')
plt.legend()

plt.show

test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()
dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:,0])


dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = np.array(y_train).flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:,0])


plt.figure(figsize=(20, 10))
plt.plot(new_y_test, label = 'Actual Close')
plt.plot(test_predictions, label = 'Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()

plt.show


