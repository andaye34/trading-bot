import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):
	df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
	df.reset_index(inplace=True)
	return df
