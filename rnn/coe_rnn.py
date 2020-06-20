import numpy as np
import pandas as pd
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
learn_end = int(end * 0.923)
x_train= xnp[ 0 : learn_end - 1 ,]
x_test=xnp[ learn_end : end-1 ,]
#每一行的y，实际上是上一行的x需要预测的值。
y_train=ynp[ 1 : learn_end ]
y_test =ynp[ learn_end + 1 : end]

x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(SimpleRNN(16,activation='tanh',input_shape=(4,1),return_sequences=True))
model.add(Dropout(0.3))
model.add(SimpleRNN(16,activation='tanh',return_sequences=True))
model.add(Dropout(0.3))
model.add(SimpleRNN(16,activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='linear'))
#es = EarlyStopping(patience=24)
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mse'])
model.fit(x_train, y_train,batch_size=32,epochs=1000,validation_data=(x_test,y_test),shuffle=False,callbacks=[])

result = model.predict(x_train,batch_size=1)
print(result)
print(y_test)
print(model.summary())
print(model.weights)

print(np.square(y_train - result))
print(np.sum(np.square(y_train - result))/len(y_train))
import matplotlib.pyplot as plt
plt.plot(y_train)
plt.plot(result)
plt.show()

