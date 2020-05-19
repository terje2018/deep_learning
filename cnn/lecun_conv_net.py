from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.utils import Sequence
import numpy as np
from cnn.DataSequence import MNISTSequence

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#X = X_train[1]/255
#TX = X_test/255
#Y = to_categorical(y_train,10)[1]
#TY = to_categorical(y_test,10)

x_train = x_train.reshape(x_train.shape[0],28,28,1)/255
x_test = x_test.reshape(x_test.shape[0],28,28,1)/255
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',kernel_initializer='glorot_normal', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='glorot_normal'))
model.add(MaxPooling2D())
model.add(Conv2D(96,(3,3),activation='relu',kernel_initializer='glorot_normal'))
#model.add(MaxPooling3D(data_format='channels_last'))
model.add(Flatten())
model.add(Dense(864, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x_train,y_train,batch_size=32, epochs=16, shuffle=True,validation_data=(x_test,y_test))

