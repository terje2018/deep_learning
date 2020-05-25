import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input , decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
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

train_labels_list = to_categorical(train_labels_list,num_classes=2)
validation_labels_list = to_categorical(validation_labels_list,num_classes=2)

base_model=VGG16(include_top=False, weights='imagenet')
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dropout(0.2)(x)
x=Dense(1024,activation='relu')(x)
preds=Dense(2,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=preds)
#print(model.summary())

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x=train_imgs_scaled, y=train_labels_list, validation_data=(validation_imgs_scaled,validation_labels_list), batch_size=32, epochs=20,verbose = 1)
print(model.summary())