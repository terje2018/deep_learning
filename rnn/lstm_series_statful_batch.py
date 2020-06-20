import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
csv = pd.read_csv("doc/time_series.csv")
length = len(csv)
print(length)

x = csv.iloc[:,1:5]
y = csv.iloc[:,5]

from sklearn import preprocessing
scale_x = preprocessing.MinMaxScaler()
xnp=np.array(x).reshape((length,4))
xnp=scale_x.fit_transform(xnp)
print(xnp)

scale_y = preprocessing.MinMaxScaler()
ynp=np.array(y).reshape((length,1))
ynp=scale_y.fit_transform(ynp)
print(ynp)

end = length
learn_end = int(end * 0.7)
x_train= xnp[ 0 : learn_end - 3 ,]
x_train=x_train.reshape((len(x_train),4,1))
y_train=ynp[ 0 : learn_end - 3 , ]#2240

validate_end = int(end * 0.15)
x_validate = xnp[ learn_end : learn_end + validate_end ,]
x_validate=x_validate.reshape((len(x_validate),4,1))
y_validate =ynp[ learn_end : learn_end + validate_end ,]#480

x_test = xnp[ learn_end + validate_end : learn_end + 2*validate_end,]
x_test=x_test.reshape((len(x_test),4,1))
y_test = ynp[ learn_end + validate_end : learn_end + 2*validate_end,]#480

#print(x_test)
#print(y_test)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

#在stateful的情况下，batch size设置主要是解决批量训练，批量验证，批量测试的要求。这种场景很适合海量数据。
#通过batch size，可以将数据打包成batch，每个batch中的每一条都参与训练，并且状态对应于其在batch中的位置（index）
#当测试的时候，数据也是同样的大小形成batch做测试，由于每个index上都存在state，所以可以并行的测试，提高效率。

#batch size设置得比较大的情况适合于离线批量处理，对于实时处理，还是设置成1，这样即保留了stateful，又能实时的处理每一条数据。
b_s = 20

model = Sequential()
model.add(LSTM(32,activation='tanh',recurrent_activation='sigmoid',return_sequences=True,stateful=True,batch_input_shape=(b_s,4,1)))
model.add(Dropout(0.3))
model.add(LSTM(32,activation='tanh',stateful=True))
model.add(Dropout(0.3))
#model.add(LSTM(4,activation='tanh'))
#model.add(Dropout(0.1))
model.add(Dense(1,activation='linear'))
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mse'])
model.fit(x_train, y_train,batch_size=b_s,shuffle=True,epochs=500,validation_data=(x_validate,y_validate))
#model.reset_states()

#这类有周期性波动的时间序列的数据，使用lstm解决起来非常合适。
xcat=np.concatenate((x_train,x_validate,x_test),axis=0)
ycat=np.concatenate((y_train,y_validate,y_test),axis=0)
presult = model.predict(xcat)
eresult = model.evaluate(xcat,ycat)
print(eresult)
plt.plot(presult,'g')
plt.plot(ycat,'r')
plt.show()

"""
model.evaluate(x_test,y_test)
presult=model.predict(x_test)
#print(presult - y_test)
plt.plot(presult,'g')
plt.plot(y_test,'r')
plt.show()
"""

#model.reset_states()#对于使用stateful而言，训练出来的model也是带有state的。如果不是stateful
#那么不会有状态保持，所以同一样本，总会得到相同的结果。
#如果是stateful，除非reset state，不然不会得到相同的结果。
#因为model里面的状态会变化，除非恰好等于上一次执行前的状态，不然结果不会相同。
"""
presult=model.predict(x_train[0:20,:],batch_size=b_s)
print(presult)
presult=model.predict(x_train[0:30,:],batch_size=b_s)
print(presult)
"""
#model.reset_states()
"""
presult=model.predict(x_train[0:20,:],batch_size=b_s)
print(presult)
presult=model.predict(x_train[0:30,:],batch_size=b_s)
print(presult)
"""

"""
print(model.summary())
print(model.weights)
"""