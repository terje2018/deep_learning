from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
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
model.add(Conv2D(8,(5,5),activation='relu',kernel_initializer='glorot_normal', input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(4608, activation='relu'))
model.add(Dense(1152, activation='linear'))
model.add(Reshape((12, 12, 8)))
model.add(Dropout(0.2))

model.add(Conv2D(16,(3,3),activation='relu',kernel_initializer='glorot_normal'))
model.add(Flatten())
model.add(Dense(1600, activation='relu'))
model.add(Dense(400, activation='linear'))
model.add(Reshape((5, 5, 16)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),activation='relu',kernel_initializer='glorot_normal'))
model.add(Flatten())
model.add(Dense(576, activation='relu'))
model.add(Dense(576, activation='linear'))
model.add(Reshape((3, 3, 64)))
model.add(GlobalAveragePooling2D())

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(x_train,y_train,batch_size=32, epochs=16, shuffle=True,validation_data=(x_test,y_test))