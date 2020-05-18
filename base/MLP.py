import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,8,0.05)
y = np.sin(x)
r = np.random.randn(160)*0.2
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
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(20, activation='sigmoid', input_dim=1))
#model.add(Dropout(0.1))#如果使用了early stop，没必要使用drop out
model.add(Dense(1, activation='linear'))
adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=32, verbose=0, mode='min')#使用'val_loss'，通常情况下不需要特别指定。
## val_loss一旦出现提高，如果在patience之内不能有效降低（min模式），那么可以提前终止训练。由于随机梯度存在"概率收敛"的情况
#所以当训练样本不多的时候，这个patience值不能设置得太小。
tx = np.arange(0,8,0.5)
ty = np.sin(tx)
model.fit(x,y, epochs=10000, batch_size=32, shuffle=True, validation_data=(tx,ty),callbacks=[es])#early stopping需要配合validation来使用。

#w = model.weights;

predictions = model.predict(tx).transpose()
s = (ty - predictions)
sumt = np.sum(s**2)
print("predictions mse: "+ str(sumt/len(tx)))

score = model.evaluate(tx, ty, batch_size=8)
print("score: "+str(score))#与predictions mse计算结果相同。
model.summary()

plt.scatter(x,y)
plt.show()