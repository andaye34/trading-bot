import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_stock_data(ticker):
    """Fetches historical stock data from Yahoo Finance."""
    df = yf.download(ticker, start="2020-01-01", end="2024-01-01")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    df = df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volume': 'Volume'})
    return df

def calculate_indicators(df):
    """Calculates VWAP, SMA, EMA, and RSI."""
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Cumulative_TP_Vol'] = (df['Typical_Price'] * df['Volume']).cumsum()
    df['Cumulative_Vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_TP_Vol'] / df['Cumulative_Vol']
    
    df['SMA_180'] = df['Close'].rolling(window=180).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    delta = df['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, abs(delta), 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def find_trade_signals(df):
    """Identifies breakout signals based on the LPP strategy."""
    signals = []
    for i in range(1, len(df)):
        if df['Close'][i] > df['SMA_180'][i] and df['RSI'][i] < 70 and df['Close'][i] > df['VWAP'][i]:
            signals.append((df['Date'][i], df['Close'][i], "BUY"))
        elif df['RSI'][i] > 70:
            signals.append((df['Date'][i], df['Close'][i], "SELL"))
    return signals

def plot_signals(df, signals):
    """Plots the price chart with buy/sell signals."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    
    buy_signals = [s for s in signals if s[2] == "BUY"]
    sell_signals = [s for s in signals if s[2] == "SELL"]
    
    if buy_signals:
        plt.scatter([s[0] for s in buy_signals], [s[1] for s in buy_signals], color='green', label='Buy Signal', marker='^')
    if sell_signals:
        plt.scatter([s[0] for s in sell_signals], [s[1] for s in sell_signals], color='red', label='Sell Signal', marker='v')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Run the Strategy
ticker = "SPY"
df = get_stock_data(ticker)
df = calculate_indicators(df)
signals = find_trade_signals(df)
plot_signals(df, signals)
