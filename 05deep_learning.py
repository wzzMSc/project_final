# %%
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout,Bidirectional,GRU

np.random.seed(1)
tf.random.set_seed(1)
prediction_targets = ['BEAMS','BEAMT','BEAMT2']
df = pd.read_csv('data_pre/data_pre.csv',header=0,parse_dates=['ISODate'],index_col='ISODate')


def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

resample_method = '3H'
df_resampled = df.resample(resample_method).mean()
df_resampled.drop(labels=['FAA'],axis=1, inplace=True)
df_resampled = df_resampled.dropna(axis=0)

dir = "vis/machine/03sig_win_reg/"

for x in prediction_targets:
    if not os.path.exists(dir+x+'/'):
        os.makedirs(dir+x+'/')

n_steps_in = 12
n_steps_out = 1
n_features = 1

for target in prediction_targets:
    scaler = MinMaxScaler()
    df_resampled[target] = scaler.fit_transform(df_resampled[target].values.reshape(-1,1))

    X,y = split_sequence(df_resampled[target],n_steps_in,n_steps_out)

    split = int(len(X)*0.9)
    X = X.reshape( (X.shape[0],X.shape[1],n_features) )
    X_train,X_test = X[:split],X[split:]
    y_train,y_test = y[:split],y[split:]

    model = Sequential()

    model.add(LSTM(100,activation='relu',input_shape=(n_steps_in,n_features)))

    # model.add(LSTM(100,activation='relu',return_sequences=True,input_shape=(n_steps_in,n_features)))
    # model.add(LSTM(100,activation='relu'))

    # model.add(Bidirectional( LSTM(200,activation='relu',input_shape=(n_steps_in,n_features)) ))

    # model.add(GRU(150,activation='relu',return_sequences=True,input_shape=(n_steps_in,n_features)))
    # model.add(GRU(150,activation='relu',return_sequences=True))
    # model.add(GRU(150,activation='relu'))

    model.add(Dense(n_steps_out))
    model.compile(
    loss='mse',
    optimizer='adam'
    )
    model.fit(X_train,y_train,epochs=500,batch_size=1000,shuffle=False,verbose=1)


    y_pred = model.predict(X_test)
    # y_pred = model.predict(X_train)

    plt.figure(figsize=(100,10))
    plt.plot(range(0,len(y_test)),y_test[:,0],label='test',color='r')
    plt.plot(range(0,len(y_test)),y_pred[:,0],label='predict',color='b')
    # plt.plot(range(0,len(y_train)),y_train[:,0],label='test',color='r')
    # plt.plot(range(0,len(y_train)),y_pred[:,0],label='predict',color='b')
    plt.legend()
    plt.savefig(dir+target+'/'+target+'.png',dpi=130)
    plt.cla()
    plt.close()
    
    print(target,'MAE:',mean_absolute_error(y_test,y_pred))
    # print(target,'MAE:',mean_absolute_error(y_train,y_pred))