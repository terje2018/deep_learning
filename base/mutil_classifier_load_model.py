import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

class1_data = np.random.randn(40,2)
class1_lable = np.ones((40,1))*0
class2_data = np.random.randn(40,2) + np.array([2.5,2.5])
class2_lable = np.ones((40,1))
class3_data = np.random.randn(40,2) + np.array([2.5,-2.5])
class3_lable = np.ones((40,1))*2
x = np.append(class1_data,class2_data,axis=0)#这里需要加上axis=0，不然不能append到一个维度上
x = np.append(x,class3_data,axis=0)
y = np.append(class1_lable,class2_lable)
y = np.append(y,class3_lable)
y = to_categorical(y,num_classes=3)

model = load_model('model.hdf5')
#model.add(Dense(3, activation='softmax', input_dim=2))载入模型不能有相同名字的层，可以换个名字。
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
filepath = 'ModelWeights-{epoch:.2f}-{accuracy:.2f}.hdf5'
mc = ModelCheckpoint(filepath, monitor='accuracy', save_best_only=True)

test_class1_data = np.random.randn(10,2)
test_class2_data = np.random.randn(10,2) + np.array([2.5,2.5])
test_class3_data = np.random.randn(10,2) + np.array([2.5,-2.5])
tx = np.append(test_class1_data,test_class2_data,axis=0)
tx = np.append(tx,test_class3_data,axis=0)
test_class1_lable = np.zeros((10,3))
test_class1_lable[:,0] = 1
test_class2_lable = np.zeros((10,3))
test_class2_lable[:,1] = 1
test_class3_lable = np.zeros((10,3))
test_class3_lable[:,2] = 1
ty = np.append(test_class1_lable,test_class2_lable,axis=0)
ty = np.append(ty,test_class3_lable,axis=0)

model.fit(x,y, epochs=50, batch_size=10, shuffle=True, callbacks=[mc],validation_data=(tx,ty))

model.save('model.hdf5')