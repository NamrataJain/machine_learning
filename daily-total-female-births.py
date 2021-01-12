import keras
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
import pandas as pd
nRowsRead =  1000
#data_set = pd.read_csv("E:/ML/daily-total-female-births-CA.CSV", delimiter=',', nrows = None, parse_dates=['date'], date_parser=pd.to_datetime)
data_set = pd.read_csv("E:/ML/daily-total-female-births-CA.CSV", delimiter=',', index_col= 0)

nRow, nCol = data_set.shape
print(f'There are {nRow} rows and {nCol} columns')
print(data_set.head(5))
print(data_set.info)
print(data_set.describe())
date_range = (data_set.index > '1959-01-01') & (data_set.index < '1959-02-01')
data_set.loc['1959-01-01':'1959-02-01']
print(data_set[date_range])
print(data_set.loc['1959-01-01':'1959-02-01'])

print(data_set.loc['1959-01-24'])

hourly = pd.DataFrame()
avg = data_set['births'].div(24).mean()
stdev = data_set['births'].div(24).std()
hourly['births'] = np.random.normal(loc=avg, scale=stdev, size=(24*365)).astype(int)
hourly.index = pd.date_range('1/1/1959', periods=(24*365), freq='H')
print(hourly.loc['1959-07-04'])



from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
