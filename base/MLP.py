import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,8,0.05)
y = np.sin(x)
r = np.random.randn(160)*0.1
y = y + r

#na = np.array([x,y]).transpose()
#print(na)
#n = np.random.random((10, 3))
#print(n)

e = y - np.sin(x)
e2 = e**2
e2s = np.sum(e2)
print("total error:" + str(e2s))
print("init mse:" + str(e2s/160))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dropout
model = Sequential()
model.add(Dense(20, activation='sigmoid', input_dim=1))
model.add(Dropout(0.1))#
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=10000, batch_size=40, shuffle=True)

#w = model.weights;
tx = np.arange(0,8,0.5)
ty = np.sin(tx)
predictions = model.predict(tx).transpose()
s = (ty - predictions)
sumt = np.sum(s**2)
print("predictions mse: "+ str(sumt/len(tx)))

score = model.evaluate(tx, ty, batch_size=8)
print("score: "+str(score))
model.summary()

plt.scatter(x,y)
plt.show()