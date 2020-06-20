import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
csv = pd.read_csv("doc/COE.csv")
#print(csv.keys())
#print(csv)
data = csv.drop(["Unnamed: 0","DATE"], axis=1)
y = data["COE$"]
x = data.drop(["COE$","Open?"], axis=1)
x = x.apply(np.log)
x = pd.concat([x,data["Open?"]],axis=1)
#print(x)
#print(y)

from sklearn import preprocessing
scale_x = preprocessing.MinMaxScaler()
xnp=np.array(x).reshape((265,4))
xnp=scale_x.fit_transform(xnp)
#print(xnp)

scale_y = preprocessing.MinMaxScaler()
ynp=np.array(y).reshape((265,1))
ynp=scale_y.fit_transform(ynp)
#print(ynp)

end = 264
learn_end = int(end * 0.915)
x_train= xnp[ 0 : learn_end - 1 ,]
#每一行的y，实际上是上一行的x需要预测的值。
y_train=ynp[ 1 : learn_end ]
x_train = x_train.reshape(x_train.shape + (1,))

x_test=xnp[ learn_end : end-3 ,]
y_test =ynp[ learn_end + 1 : end-2]
x_test = x_test.reshape(x_test.shape + (1,))

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

#batch size=1，主要是为了使用stateful功能。
#如果需要批量的处理数据，这里可以设置得大一些，表示每一次都需要并行的处理一批数据。
#例如batch size=16，意味着每次都需要并行处理16条数据，这16条数据作为一个整体出现。
#测试的时候，预测的时候，也是一次16条（一个batch）
#本质上是将一个batch的数据分成16条并行处理的过程。每条数据对应在每个batch中的位置（index）。
#根据位置来确定每条数据所应对的state。
#例如index=1，每次只会改变index=1这个位置的state。
b_s = 1
model = Sequential()
model.add(LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid',return_sequences=True,stateful=True,batch_input_shape=(b_s,4,1)))
model.add(Dropout(0.3))
model.add(LSTM(32,activation='tanh',recurrent_activation='hard_sigmoid',stateful=True,batch_input_shape=(b_s,4,1)))
model.add(Dropout(0.3))
model.add(Dense(1,activation='linear'))
#es = EarlyStopping(patience=24)
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mse'])
model.fit(x_train, y_train,batch_size=b_s,shuffle=True,epochs=1000)
#model.reset_states()

result = model.evaluate(x_test,y_test)
print(result)

"""
presult=model.predict(x_test)
print(presult - y_test)
plt.plot(presult,'r')
plt.plot(y_test,'g')
plt.show()
"""

#对于coe数据的拟合，比没有stateful的simple cnn要好一些。
xnp = xnp.reshape(xnp.shape + (1,))
result = model.evaluate(xnp,ynp)
print(result)
presult=model.predict(xnp)
print(presult - ynp)
plt.plot(presult,'r')
plt.plot(ynp,'g')
plt.show()