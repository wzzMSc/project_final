# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error,mean_absolute_error
import statsmodels.api as sm


prediction_targets = ['BEAMS','BEAMT','BEAMT2']

# %%
dir = "vis/stat/01naive/"

df = pd.read_csv('data_pre/data_pre.csv',header=0,parse_dates=['ISODate'],index_col='ISODate')

resample_method = '3H'
df_resampled = df.resample(resample_method).mean()
df_resampled.drop(labels=['FAA'],axis=1, inplace=True)
df_resampled = df_resampled.dropna(axis=0)
df_resampled['i'] = [x for x in range(1,len(df_resampled)+1)]

for x in prediction_targets:
    if not os.path.exists(dir+x+'/'):
        os.makedirs(dir+x+'/')

s = int(len(df_resampled)*0.8)
train,test = df_resampled[:s],df_resampled[s:]

for y in prediction_targets:
    pred = [train[y][-1]]*len(test)
    plt.figure(figsize=(150,10))
    plt.plot(train.i,train[y],label='Train',color='r')
    plt.plot(test.i,test[y],label='Test',color='g')
    plt.plot(test.i,pred,label='Predict',color='b')
    plt.xlabel('Sequencial entries')
    plt.ylabel(y)
    plt.legend()
    plt.savefig(dir+y+'/'+y+'.png',dpi=130)
    plt.cla()
    plt.close()

    print(y,'MAE:',mean_absolute_error(test[y],pred))

# %%
dir = "vis/stat/02naive_mean/"

for x in prediction_targets:
    if not os.path.exists(dir+x+'/'):
        os.makedirs(dir+x+'/')


for y in prediction_targets:
    pred = [train[y].mean()]*len(test)
    plt.figure(figsize=(150,10))
    plt.plot(train.i,train[y],label='Train',color='r')
    plt.plot(test.i,test[y],label='Test',color='g')
    plt.plot(test.i,pred,label='Predict',color='b')
    plt.xlabel('Sequencial entries')
    plt.ylabel(y)
    plt.legend()
    plt.savefig(dir+y+'/'+y+'.png',dpi=130)
    plt.cla()
    plt.close()

    print(y,'MAE:',mean_absolute_error(test[y],pred))

# %%
dir = "vis/stat/03rolling_mean/"



for x in prediction_targets:
    if not os.path.exists(dir+x+'/'):
        os.makedirs(dir+x+'/')


for y in prediction_targets:
    pred = [train[y].rolling(10).mean()[-1]]*len(test)
    plt.figure(figsize=(150,10))
    plt.plot(train.i,train[y],label='Train',color='r')
    plt.plot(test.i,test[y],label='Test',color='g')
    plt.plot(test.i,pred,label='Predict',color='b')
    plt.xlabel('Sequencial entries')
    plt.ylabel(y)
    plt.legend()
    plt.savefig(dir+y+'/'+y+'.png',dpi=130)
    plt.cla()
    plt.close()

    print(y,'MAE:',mean_absolute_error(test[y],pred))

# %%
dir = "vis/stat/04single_exp/"


for x in prediction_targets:
    if not os.path.exists(dir+x+'/'):
        os.makedirs(dir+x+'/')


for y in prediction_targets:
    ses = SimpleExpSmoothing(train[y].values).fit()
    test['SES'] = ses.forecast(len(test))
    plt.figure(figsize=(150,10))
    plt.plot(train.i,train[y],label='Train',color='r')
    plt.plot(test.i,test[y],label='Test',color='g')
    plt.plot(test.i,test['SES'],label='Predict',color='b')
    plt.xlabel('Sequencial entries')
    plt.ylabel(y)
    plt.legend()
    plt.savefig(dir+y+'/'+y+'.png',dpi=130)
    plt.cla()
    plt.close()

    print(y,'MAE:',mean_absolute_error(test[y],test['SES']))
# %%
dir = "vis/stat/05holt/"

for x in prediction_targets:
    if not os.path.exists(dir+x+'/'):
        os.makedirs(dir+x+'/')

for y in prediction_targets:
    sm.tsa.seasonal_decompose(train[y].values,period=8*30).plot(resid=False)
    plt.savefig(dir+y+'/'+y+'_seasonal_decompose.png',dpi=130)
    plt.cla()
    plt.close()

for y in prediction_targets:
    holt = Holt(train[y].values).fit()
    test['holt'] = holt.forecast(len(test))
    plt.figure(figsize=(150,10))
    plt.plot(train.i,train[y],label='Train',color='r')
    plt.plot(test.i,test[y],label='Test',color='g')
    plt.plot(test.i,test['holt'],label='Predict',color='b')
    plt.xlabel('Sequencial entries')
    plt.ylabel(y)
    plt.legend()
    plt.savefig(dir+y+'/'+y+'.png',dpi=130)
    plt.cla()
    plt.close()

    print(y,'MAE:',mean_absolute_error(test[y],test['holt']))
# %%
dir = "vis/stat/06holt_winter/"

for x in prediction_targets:
    if not os.path.exists(dir+x+'/'):
        os.makedirs(dir+x+'/')


for y in prediction_targets:
    es = ExponentialSmoothing(train[y].values,seasonal_periods=8*30,trend='add',seasonal='add').fit()
    test['es'] = es.forecast(len(test))
    plt.figure(figsize=(150,10))
    plt.plot(train.i,train[y],label='Train',color='r')
    plt.plot(test.i,test[y],label='Test',color='g')
    plt.plot(test.i,test['es'],label='Predict',color='b')
    plt.xlabel('Sequencial entries')
    plt.ylabel(y)
    plt.legend()
    plt.savefig(dir+y+'/'+y+'.png',dpi=130)
    plt.cla()
    plt.close()

    print(y,'MAE:',mean_absolute_error(test[y],test['es']))

# %%
