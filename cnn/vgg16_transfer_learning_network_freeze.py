from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from keras.utils.np_utils import to_categorical
import numpy as np
import glob

input_shape = (150, 150, 3)

train_files = glob.glob('images/dogs_and_cats/*')
train_imgs = [image.img_to_array(image.load_img(img, target_size=input_shape)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('/')[2].split('.')[0].strip() for fn in train_files]

validation_files = glob.glob('images/dogs_and_cats_validation/*')
validation_imgs = [image.img_to_array(image.load_img(img, target_size=input_shape)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('/')[2].split('.')[0].strip() for fn in validation_files]

train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

train_labels_list = []
for l in train_labels:
    if l == 'dog':
        train_labels_list.append(1)
    else:
        train_labels_list.append(0)

validation_labels_list = []
for l in validation_labels:
    if l == 'dog':
        validation_labels_list.append(1)
    else:
        validation_labels_list.append(0)

train_labels_list = np.array(train_labels_list)
validation_labels_list = np.array(validation_labels_list)
#train_labels_list = to_categorical(train_labels_list,num_classes=2)
#validation_labels_list = to_categorical(validation_labels_list,num_classes=2)

vgg=VGG16(include_top=False, weights='imagenet',input_shape=input_shape)
output = vgg.layers[-1].output
output = Flatten()(output)
vgg_model = Model(vgg.input, output)
#vgg_model.trainable = False
#for layer in vgg_model.layers:
#    layer.trainable = False
input_shape = vgg_model.output_shape[1]


def get_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features


train_features_vgg = get_features(vgg_model, train_imgs_scaled)
validation_features_vgg = get_features(vgg_model, validation_imgs_scaled)

model = Sequential()
model.add(InputLayer(input_shape=(input_shape,)))
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=train_features_vgg, y=train_labels_list, validation_data=(validation_features_vgg,validation_labels_list), batch_size=32, epochs=10,verbose = 1)
print(model.summary())