import numpy as np
np.random.seed(1)
import tensorflow as tf
from tensorflow import random
tf.compat.v1.random.set_random_seed(2)
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import datetime as dt
import time, warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

plt.style.use('ggplot')

url = 'BTC.csv'
df = pd.read_csv(url, parse_dates=True, index_col=0)

df.corr()['Price']

earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=80,  verbose=1, mode='min')
callbacks_list = [earlystop]

def fit_model(train, val, timesteps, hl, lr, batch, epochs):
    X_train, Y_train, X_val, Y_val = [], [], [], []

    for i in range(timesteps, train.shape[0]):
        X_train.append(train[i-timesteps:i])
        Y_train.append(train[i][0])
    X_train, Y_train = np.array(X_train), np.array(Y_train)

    for i in range(timesteps, val.shape[0]):
        X_val.append(val[i-timesteps:i])
        Y_val.append(val[i][0])
    X_val, Y_val = np.array(X_val), np.array(Y_val)
    
    model = Sequential()
    model.add(LSTM(X_train.shape[2], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True,
                   activation = 'relu'))

    for i in range(len(hl)-1):        
        model.add(LSTM(hl[i], activation='relu', return_sequences=True))

    model.add(LSTM(hl[-1], activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(lr=lr), loss='mean_squared_error')

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, validation_data=(X_val, Y_val), 
    					verbose=1, shuffle=False, callbacks=callbacks_list)

    model.reset_states()
    return model, history.history['loss'], history.history['val_loss']

def evaluate_model(model, test, timesteps):
    X_test, Y_test = [], []

    for i in range(timesteps, test.shape[0]):
        X_test.append(test[i-timesteps:i])
        Y_test.append(test[i][0])
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    Y_hat = model.predict(X_test)
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))

    #Y_hat = np.concatenate((Y_hat, X_test[:, 1:]), axis=1)
    #Y_hat = sc.inverse_transform(Y_hat)
    #Y_hat = Y_hat[:, 0]

    #Y_test = Y_test.reshape((len(Y_test), 1))
    #Y_test = np.concatenate((Y_test, X_test[:, 1:]), axis=1)
    #Y_test = sc.inverse_transform(Y_test)
    #Y_test = Y_test[:, 0]

    mse = mean_squared_error(Y_test, Y_hat)
    rmse = sqrt(mse)
    r = r2_score(Y_test, Y_hat)
    return mse, rmse, r, Y_test, Y_hat

def plot_data(Y_test, Y_hat):
    plt.plot(Y_test, c='r')
    plt.plot(Y_hat, c='y')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Stock Prediction Graph using Multivariate-LSTM model')
    plt.legend(['Actual', 'Predicted'], loc = 'lower right')
    plt.show()

series = df[['Price', 'High', 'Low']] 

dates = df.index.to_pydatetime()
train_dates = dates[0:int(len(dates)/2)+int(len(dates)/8)]
val_dates = dates[int(len(train_dates)):int(len(train_dates))+int(len(dates)/16)]
test_dates = dates[int(len(train_dates))+int(len(dates)/16):-1]

train_start = train_dates[0]
train_end = train_dates[-1]
train_data = series.loc[train_start:train_end]
train_data = np.where(train_data=='—', 0, train_data)

val_start = val_dates[0]
val_end = val_dates[-1]
val_data = series.loc[val_start:val_end]
val_data = np.where(val_data=='—', 0, val_data)

test_start = test_dates[0]
test_end = dt.datetime.now() + dt.timedelta(days=1)
test_data = series.loc[test_start:test_end]
test_data = np.where(test_data=='—', 0, test_data)

sc = MinMaxScaler()
train = sc.fit_transform(train_data)
val = sc.fit_transform(val_data)
test = sc.fit_transform(test_data)

timesteps = 1
hl = [40, 35]
lr = 1e-3
batch_size = 256
num_epochs = 250
model, train_error, val_error = fit_model(train, val, timesteps, hl, lr, batch_size, num_epochs)

mse, rmse, r2_value, true, predicted = evaluate_model(model, test, timesteps)
predicted = sc.inverse_transform(predicted)

plot_data(true, predicted)

print(predicted)

#model.save('lecture13.LSTM_3_stock_multivariate.h5')

#del model # Deletes the model
# Load a model
#model = load_model('MV3-LSTM_50_[40,35]_1e-3_64.h5')