from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Reshape
from keras.layers.pooling import GlobalAveragePooling2D
from keras.utils import Sequence
import numpy as np
from cnn.DataSequence import MNISTSequence

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1)/255
x_test = x_test.reshape(x_test.shape[0],28,28,1)/255
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

model = Sequential()
model.add(Conv2D(8,(3,3),activation='relu',kernel_initializer='glorot_normal', input_shape=(28,28,1)))
model.add(Conv2D(8,(3,3),activation='relu',kernel_initializer='glorot_normal'))
model.add(MaxPooling2D())

model.add(Conv2D(16,(3,3),activation='relu',kernel_initializer='glorot_normal'))
model.add(Conv2D(16,(3,3),activation='relu',kernel_initializer='glorot_normal'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#这里的accuracy是没一个batch体现出来的指标

print(model.summary())
filepath = 'vgg_conv_net-ModelWeights-{epoch:.2f}-{accuracy:.2f}.hdf5'
#save_best_only=True, mode='max'通常不设置，因为每一轮训练结果的最优，不代表在测试数据集上表现也好
#提高训练的epochs次数，通过checkpoint callback获取到多个训练结果，之后从这些结果中选一个最优解。
mc = ModelCheckpoint(filepath, monitor='accuracy')
model.fit(x_train,y_train,batch_size=32, epochs=32, shuffle=True,validation_data=(x_test,y_test),callbacks=[mc])