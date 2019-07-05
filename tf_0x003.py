# -*- coding:utf-8 -*- 
# @Author: Jone Chiang
# @Date  : 2019/7/4 15:10
# @File  : tf_0x003

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

fasion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fasion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


# train_images = train_images / 255.0
# test_images = test_images / 255.0
#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy： {}'.format(test_acc))

predictions = model.predict(test_images)

