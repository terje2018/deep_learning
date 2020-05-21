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
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],28,28,1)/255
x_test = x_test.reshape(x_test.shape[0],28,28,1)/255
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

for index in range(16,32,1):
    model = load_model('vgg_conv_net-ModelWeights-'+str(index)+'.00-1.00.hdf5')  # 提供了一种分阶段训练的思路。
    result = model.evaluate(x_test, y_test)
    print(result)

#model.fit(x_train,y_train, batch_size=32, epochs=24, shuffle=True,validation_data=(x_test,y_test))