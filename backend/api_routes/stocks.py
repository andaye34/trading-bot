from fastapi import FastAPI
import yfinance as yf
import torch
import numpy as np
from model import load_model
from train import scaler, lookback

app = FastAPI()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = load_model()
model.to(device)
model.eval()

def get_stock_prediction(ticker: str):
    data = yf.download(ticker, period='60d')[['Close']]
    if len(data) < lookback:
        return {"error": "Not enough historical data"}
    
    last_lookback = data['Close'].values[-lookback:].reshape(1, lookback, 1)
    last_lookback_scaled = scaler.transform(last_lookback.reshape(lookback, 1)).reshape(1, lookback, 1)
    X_input = torch.tensor(last_lookback_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        predicted_scaled = model(X_input).cpu().numpy()
    
    predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0, 0]
    last_close = data['Close'].iloc[-1]
    change_percent = (predicted_price - last_close) / last_close * 100
    
    sentiment = "bullish" if change_percent > 1 else "bearish" if change_percent < -1 else "neutral"
    
    return {
        "ticker": ticker,
        "last_close": last_close,
        "predicted_close": predicted_price,
        "percent_change": round(change_percent, 2),
        "sentiment": sentiment
    }

@app.get("/predict/{ticker}")
def predict_stock(ticker: str):
    return get_stock_prediction(ticker)

