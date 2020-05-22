from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.pooling import MaxPooling3D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Reshape
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.pooling import GlobalAveragePooling3D
from keras.utils import Sequence
import numpy as np
from cnn.DataSequence import MNISTSequence
from keras.layers import concatenate
from keras.layers import Input
from keras.models import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1)/255
x_test = x_test.reshape(x_test.shape[0],28,28,1)/255
y_train = to_categorical(y_train,num_classes=10)
y_test = to_categorical(y_test,num_classes=10)

input_img = Input(shape = (28,28,1))
tower1_1 = Conv2D(24,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(input_img)
tower1_1 = Conv2D(24,(3,3),activation='relu',padding='same',kernel_initializer='glorot_normal')(tower1_1)
tower2_1 = Conv2D(24,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(input_img)
tower2_1 = Conv2D(24,(5,5),activation='relu',padding='same',kernel_initializer='glorot_normal')(tower2_1)
tower3_1 = MaxPooling2D((3,3),strides=(1,1), padding='same')(input_img)
tower3_1 = Conv2D(24,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(tower3_1)
tower4_1 = Conv2D(24,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(input_img)
output1 = concatenate([tower1_1,tower2_1,tower3_1,tower4_1],axis=3)
output1 = MaxPooling2D()(output1)

tower1_2 = Conv2D(16,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(output1)
tower1_2 = Conv2D(16,(3,3),activation='relu',padding='same',kernel_initializer='glorot_normal')(tower1_2)
tower2_2 = Conv2D(16,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(output1)
tower2_2 = Conv2D(16,(5,5),activation='relu',padding='same',kernel_initializer='glorot_normal')(tower2_2)
tower3_2 = MaxPooling2D((3,3),strides=(1,1), padding='same')(output1)
tower3_2 = Conv2D(16,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(tower3_2)
tower4_2 = Conv2D(16,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(output1)
output2 = concatenate([tower1_2,tower2_2,tower3_2,tower4_2],axis=3)
output2 = MaxPooling2D()(output2)

#tower1_3 = Conv2D(8,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(output2)
#tower1_3 = Conv2D(8,(3,3),activation='relu',padding='same',kernel_initializer='glorot_normal')(tower1_3)
#tower2_3 = Conv2D(8,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(output2)
#tower2_3 = Conv2D(8,(5,5),activation='relu',padding='same',kernel_initializer='glorot_normal')(tower2_3)
#tower3_3 = MaxPooling2D((3,3),strides=(1,1), padding='same')(output2)
#tower3_3 = Conv2D(8,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(tower3_3)
#tower4_3 = Conv2D(8,(1,1),activation='relu',padding='same',kernel_initializer='glorot_normal')(output2)
#output3 = concatenate([tower1_3,tower2_3,tower3_3,tower4_3],axis=3)
#output3 = MaxPooling2D()(output3)

#output4 = GlobalAveragePooling2D()(output2)#只有层非常深的时候，全局平均才有意义。
output3 = Flatten()(output2)#层不够深，直接flatten就可以了
output4 = Dense(1568, activation='relu')(output3)
output4 = Dropout(0.2)(output4)
output4 = Dense(784, activation='relu')(output3)
output4 = Dropout(0.2)(output4)
output5 = Dense(10, activation='softmax')(output4)

model = Model(inputs=input_img, outputs=output5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#这里的accuracy是没一个batch体现出来的指标
print(model.summary())
filepath = 'googlenet_conv_net-ModelWeights-{epoch:.2f}-{accuracy:.2f}.hdf5'
#save_best_only=True, mode='max'通常不设置，因为每一轮训练结果的最优，不代表在测试数据集上表现也好
#提高训练的epochs次数，通过checkpoint callback获取到多个训练结果，之后从这些结果中选一个最优解。
mc = ModelCheckpoint(filepath, monitor='accuracy')
model.fit(x_train,y_train,batch_size=32, epochs=32, shuffle=True,validation_data=(x_test,y_test),callbacks=[mc])