import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
xls = pd.ExcelFile('doc/stockindexes.xls')
print(xls.sheet_names)

ftse100 = pd.read_excel(xls,'FTSE100')
dow_jones = pd.read_excel(xls,'Dow Jones Industrial')

ftse100_data = ftse100.iloc[4:1357,1]
dow_jones_data = dow_jones.iloc[4:1357,1]

print(ftse100_data.head())
print(dow_jones_data.head())

y2 = pd.concat([ftse100_data,dow_jones_data],axis=1)
print(y2.head())
y2 = y2.reset_index(drop=True)
print(y2.head())
y2.columns = ['ftse100','dow_jones']
print(y2.head())

y2 = y2.pct_change(1)
print(y2.head())
#这里的center=t，配合window来使用，统计窗口在中间的位置，所以最前面的15条和最后的15条都是nan，因为窗口滑不到那里。，
vol_y2 = y2.rolling(window=30,center=True).std()#标准差.
"""
print(vol_y2)
plt.plot(vol_y2)
plt.show()
"""
x1 = np.log((vol_y2.shift(1)/vol_y2.shift(2))*vol_y2.shift(1))
x2 = np.log((vol_y2.shift(1)/vol_y2.shift(3))*vol_y2.shift(1))
x3 = np.log((vol_y2.shift(1)/vol_y2.shift(4))*vol_y2.shift(1))
x4 = np.log((vol_y2.shift(1)/vol_y2.shift(5))*vol_y2.shift(1))
x5 = np.log((vol_y2.shift(1)/vol_y2.shift(6))*vol_y2.shift(1))

data = pd.concat([vol_y2,x1,x2,x3,x4,x5],axis=1)
data.columns = ['ftse100','dow_jones','ftse100_1','dow_jones_1','ftse100_2','dow_jones_2','ftse100_3','dow_jones_3','ftse100_4','dow_jones_4','ftse100_5','dow_jones_5']
data = data.dropna()
y = data[['ftse100','dow_jones']]
x = data[['ftse100_1','dow_jones_1','ftse100_2','dow_jones_2','ftse100_3','dow_jones_3','ftse100_4','dow_jones_4','ftse100_5','dow_jones_5']]
print(y)
print(x)

scaler_x = preprocessing.MinMaxScaler(feature_range=(0,1))
x = np.array(x).reshape((len(x),10))
x = scaler_x.fit_transform(x)
x = np.array(x).reshape((len(x),10,1))
scaler_y = preprocessing.MinMaxScaler(feature_range=(0,1))
y = np.array(y).reshape((len(y),2))
y = scaler_y.fit_transform(y)
#y = np.array(y).reshape((len(y),2,1))

x_train = x[0:1000,:]
y_train = y[0:1000,:]
x_validate = x[1000:1150,:]
y_validate = y[1000:1150,:]
x_test = x[1150:1317,:]
y_test = y[1150:1317,:]

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#两类数据去拟合两类结果，做时间序列上的拟合。
#这里隐含着一个意思，通过把两类数据合并在一起拟合，可以把他们之间彼此存在的关系给拟合出来。
b_s = 1
model = Sequential()
model.add(LSTM(32,activation='tanh',recurrent_activation='sigmoid',batch_input_shape=(b_s,10,1)))
model.add(Dropout(0.2))
#model.add(LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid',stateful=True,batch_input_shape=(b_s,4,1)))
#model.add(Dropout(0.3))
model.add(Dense(2,activation='linear'))#输出维度为2，可以认为在计算平面上的距离。
#es = EarlyStopping(patience=24)
print(model.summary())
print(model.weights)
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mse'])
model.fit(x_train, y_train,batch_size=b_s,shuffle=True,epochs=30)
#model.reset_states()

result = model.evaluate(x_test,y_test)
print(result)
presult = model.predict(x_test)
#print(presult)

result = model.predict(x_test)
#print(np.square(y_test - result))
#print(np.sum(np.square(y_test[:,0] - result[:,0])))
#print(np.sum(np.square(y_test[:,1] - result[:,1])))
print(np.sum(np.square(y_test - result))/(y_test.shape[0]*y_test.shape[1]))

plt.plot(y_test[:,1],'r')
plt.plot(result[:,1],'g')
plt.show()