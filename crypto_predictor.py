import numpy as np
import tensorflow as tf
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow import random
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

plt.style.use('ggplot')

np.random.seed(1)

earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')
callbacks_list = [earlystop]

crypto_list = ['BTC.csv', 'ETH.csv', 'SHIB.csv']

def fit_model(train, val, timesteps, hl, lr, batch, epochs):
	X_train, Y_train, X_val, Y_val = [], [], [], []

	for i in range(timesteps, train.shape[0]):
		X_train.append(train[i-timesteps:i])
		Y_train.append(train[i])
	X_train, Y_train = np.array(X_train), np.array(Y_train)

	for i in range(timesteps, val.shape[0]):
		X_val.append(val[i-timesteps:i])
		Y_val.append(val[i])
	X_val, Y_val = np.array(X_val), np.array(Y_val)
  
	model = Sequential()
	model.add(LSTM(10, input_shape=(X_train.shape[1], 1), return_sequences=True, activation='relu'))
	for i in range(len(hl)-1):        
		model.add(LSTM(hl[i], activation='relu', return_sequences=True))
	model.add(LSTM(hl[-1], activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer=optimizers.Adam(lr=lr), loss='mean_squared_error')
  
	history = model.fit(X_train, Y_train, epochs = epochs, batch_size = batch, validation_data = (X_val, Y_val),
						verbose = 1, shuffle = False, callbacks=callbacks_list)
	model.reset_states()
	return model, history.history['loss'], history.history['val_loss']

def evaluate_model(model, test, timesteps):
	X_test = []
	Y_test = []

	for i in range(timesteps, test.shape[0]):
		X_test.append(test[i-timesteps:i])
		Y_test.append(test[i])
	X_test, Y_test = np.array(X_test), np.array(Y_test)
	  
	Y_hat = model.predict(X_test)
	mse = mean_squared_error(Y_test, Y_hat)
	rmse = sqrt(mse)
	r2 = r2_score(Y_test, Y_hat)
	return rmse, r2, Y_test, Y_hat  

def plot_data(Y_test, Y_hat):
	plt.plot(Y_test, c='r')
	plt.plot(Y_hat, c='y')
	plt.xlabel('Day')
	plt.ylabel('Price')
	plt.title("Bitcoin Price Prediction")
	plt.legend(['Actual', 'Predicted'], loc = 'lower right')
	plt.show()

for crypto_name in crypto_list:
	url = crypto_name
	df = pd.read_csv(crypto_name, parse_dates=True, index_col='Date')
	df.index.to_pydatetime()
	series = df['Price']

	dates = df.index.to_pydatetime()
	train_dates = dates[0:int(len(dates)/2)+int(len(dates)/8)]
	val_dates = dates[int(len(train_dates)):int(len(train_dates))+int(len(dates)/16)]
	test_dates = dates[int(len(train_dates))+int(len(dates)/16):-1]

	train_start = train_dates[0]
	train_end = train_dates[-1]
	train_data = series.loc[train_start:train_end].values.reshape(-1, 1)
	train_data = np.where(train_data=='—', 0, train_data)

	val_start = val_dates[0]
	val_end = val_dates[-1]
	val_data = series.loc[val_start:val_end].values.reshape(-1, 1)
	val_data = np.where(val_data=='—', 0, val_data)

	test_start = test_dates[0]
	test_end = dt.datetime.now() + dt.timedelta(days=1)
	test_data = series.loc[test_start:test_end].values.reshape(-1, 1)
	test_data = np.where(test_data=='—', 0, test_data)

	sc = MinMaxScaler()
	train = sc.fit_transform(train_data)
	val = sc.fit_transform(val_data)
	test = sc.fit_transform(test_data)

	timesteps = 1
	hl = [40, 35]
	lr = 1e-3
	batch_size = 64
	num_epochs = 500

	model, train_error, val_error = fit_model(train, val, timesteps, hl, lr, batch_size, num_epochs)

	rmse, r2_value, true, predicted = evaluate_model(model, test, timesteps)

	print(predicted)

	true = sc.inverse_transform(true).reshape(-1, 50)
	predicted = sc.inverse_transform(predicted).reshape(-1, 50)

	plot_data(true, predicted)
	prices_df = pd.DataFrame(predicted[-1], columns=['Price Tomorrow'], index=[crypto_name.strip('.csv')])
	prices_df.to_csv('All.csv', mode='a+', header=False)

	print('Predicted value for December 31, 2021: ${:0.2f}'.format(predicted[-1][0]))