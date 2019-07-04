# -*- coding:utf-8 -*- 
# @Author: Jone Chiang
# @Date  : 2019/7/4 11:17
# @File  : tf_0x002
import os
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略cpu相关的warning

Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Conv2D = keras.layers.Conv2D
Model = keras.Model


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()
loss_object = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = keras.metrics.Mean(name='test_loss')
test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5
for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_iamges, test_labels in test_ds:
        test_step(test_iamges, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))


"""
result:

Epoch 1, Loss: 0.1363905817270279, Accuracy: 95.86166381835938, Test Loss: 0.06474881619215012, Test Accuracy: 97.87999725341797
Epoch 2, Loss: 0.08842511475086212, Accuracy: 97.30999755859375, Test Loss: 0.060913700610399246, Test Accuracy: 97.9749984741211
Epoch 3, Loss: 0.06511753797531128, Accuracy: 98.02055358886719, Test Loss: 0.06148631125688553, Test Accuracy: 98.00999450683594
Epoch 4, Loss: 0.05195368081331253, Accuracy: 98.41333770751953, Test Loss: 0.061374157667160034, Test Accuracy: 98.10250091552734
Epoch 5, Loss: 0.043499212712049484, Accuracy: 98.66633605957031, Test Loss: 0.06273096054792404, Test Accuracy: 98.15399932861328
"""