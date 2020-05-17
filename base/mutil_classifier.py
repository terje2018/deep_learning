import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

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

model = Sequential()
model.add(Dense(3, activation='softmax', input_dim=2))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=5000, batch_size=20, shuffle=True)

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
score = model.evaluate(tx, ty, batch_size=1)
print("score： " + str(score))
probs = model.predict_proba(tx)
#print("predict_proba: " + str(probs))


def cross_entropy(probs,labels):
    length = len(labels)
    ce = 0
    for i in range(0,length):
        label = labels[i,:]
        index = (label == 1)
        p = probs[i,index]
        ce = ce - np.log(p)
    return ce/length


pty = model.predict_classes(tx)
print(pty)
print("cross_entropy: "+ str(cross_entropy(probs,ty)))
model.summary()

plt.scatter(class1_data[:,0],class1_data[:,1])
plt.scatter(class2_data[:,0],class2_data[:,1])
plt.scatter(class3_data[:,0],class3_data[:,1])
plt.show()