# predict.py - Loads trained model and makes predictions

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import load_model
from train import scaler, lookback, X_test, y_test

#set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#load the trained model
model = load_model()
model.to(device)
model.eval()
#make predictions
with torch.no_grad():
    predicted = model(X_test.to(device)).cpu().numpy()


#reshape predicted values for inverse transformation
dummies = np.zeros((predicted.shape[0], lookback + 1))
dummies[:, 0] = predicted.flatten()
dummies = scaler.inverse_transform(dummies)
predicted_prices = dummies[:, 0]

# Convert actual y_test back to original scale
y_test_reshaped = y_test.reshape(-1,1) #making sure its a 2d array before inverse transformation
#Ensure the shape matches what the scaler expects
y_test_padded = np.zeros((y_test.shape[0], lookback + 1)) #shape 198,21
y_test_padded[:, 0] = y_test.flatten() #only modify first column
#inverse transform using minmax
actual_prices = scaler.inverse_transform(y_test_padded)[:, 0] #extract only the first column

##############################
#generate sbuy/sell signals on a 1% threshold
def get_signal(prices):
	signals = ["HOLD"]  #first day has no previous price for comparison
	for i in range(1, len(prices)):
		pct_change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
		if pct_change >0.5: #1%
			signals.append("BUY")
		elif pct_change < -0.5: # more than -1% decreas
			signals.append("SELL")
		else:
			signals.append("HOLD")
	return signals
#generate signals for actual  and predicted prices
actual_signals = get_signal(actual_prices)
predicted_signals = get_signal(predicted_prices)

# Print first 10 results for comparison
print("\nFirst 10 Predictions vs Actuals with Buy/Sell Signals:")
print(f"{'Day':<5} {'Actual Price':<15} {'Actual Signal':<10} {'Predicted Price':<15} {'Predicted Signal':<10}")
for i in range(10):
    print(f"{i:<5} {actual_prices[i]:<15.2f} {actual_signals[i]:<10} {predicted_prices[i]:<15.2f} {predicted_signals[i]:<10}")




plt.figure(figsize=(20, 10))
plt.plot(actual_prices, label='Actual Close')
plt.plot(predicted_prices, label='Predicted Close')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.savefig("prediction_plot.png")
