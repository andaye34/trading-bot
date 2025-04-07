import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def prepare_dataframe(df, target_column, lookback):
	df_processed = df[[target_column]].copy()
	for i in range(1, lockback+1):
		df_processed[f'{target_column}(t-{i})'] = df_processed[target_column].shift(i)
	df_processed.dropna(inplace=True)
	return df_processed

def scaled_data(df)
	scaler = MinMaxScaler((-1,1))
	scaled = scaler.fit_transform(df.values)
	return scaled, scaler

