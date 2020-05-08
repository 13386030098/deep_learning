import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Regressor
# X = np.linspace(-1, 1, 200)
# np.random.shuffle(X)    # randomize the data
# Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# # plt.scatter(X, Y)
# # plt.show()
# X_train, Y_train = X[:160], Y[:160]     # first 160 data points
# X_test, Y_test = X[160:], Y[160:]       # last 40 data points
#
#
#
# model = keras.Sequential()
# model.add(layers.Dense(units=1, input_dim=1))
# # choose loss function and optimizing method
# model.compile(loss='mse', optimizer='sgd')
# # training
# print('Training -----------')
# for step in range(5000):
#     cost = model.train_on_batch(X_train, Y_train)
#     if step % 100 == 0:
#         print('train cost: ', cost)
#
# # test
# print('\nTesting ------------')
# cost = model.evaluate(X_test, Y_test, batch_size=40)
# print('test cost:', cost)
# W, b = model.layers[0].get_weights()
# print('Weights=', W, '\nbiases=', b)
#
# Y_pred = model.predict(X_test)
# plt.scatter(X_test, Y_test)
# plt.plot(X_test, Y_pred)
# plt.show()

# Classifier

# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# # data pre-processing
# X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
# X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
# y_train = keras.utils.to_categorical(y_train, num_classes=10)
# y_test = keras.utils.to_categorical(y_test, num_classes=10)
#
# model = keras.models.Sequential([
#     keras.layers.Dense(32, input_dim=784),
#     keras.layers.Activation('relu'),
#     keras.layers.Dense(10),
#     keras.layers.Activation('softmax'),
# ])
#
# # Another way to define your optimizer
# rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#
# # We add metrics to get more results you want to see
# model.compile(optimizer=rmsprop,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(X_train, y_train, epochs=2, batch_size=32)
#
# loss, accuracy = model.evaluate(X_test, y_test)
#
# print('test loss: ', loss)
# print('test accuracy: ', accuracy)


#Convolutional Neural Network
# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# # data pre-processing
# X_train = X_train.reshape(-1, 1,28, 28)/255.
# X_test = X_test.reshape(-1, 1,28, 28)/255.
# y_train = keras.utils.to_categorical(y_train, num_classes=10)
# y_test = keras.utils.to_categorical(y_test, num_classes=10)
#
# # keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
# # keras.models import Sequential
# # Another way to build your CNN
# model = keras.models.Sequential([
#     keras.layers.Conv2D(
#         batch_input_shape=(None, 1, 28, 28),
#         filters=32,
#         kernel_size=5,
#         strides=1,
#         padding='same',  # Padding method
#         data_format='channels_first',
#         activation='relu',
#     ),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Dropout(0.2),
#     keras.layers.Conv2D(64, 5, strides=1, padding='same', data_format='channels_first',activation='relu',),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Flatten(),
#     keras.layers.Dense(1024, activation='relu'),
#     keras.layers.Dense(10, activation='softmax'),
# ])
# # Conv layer 1 output shape (32, 28, 28)
#
# adam = keras.optimizers.Adam(lr=1e-4)
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# print('Training ------------')
# # Another way to train the model
# model.fit(X_train, y_train, epochs=1, batch_size=64,)
#
# loss, accuracy = model.evaluate(X_test, y_test)
# print('\ntest loss: ', loss)
# print('\ntest accuracy: ', accuracy)


# Recurrent Neural Network (long-short term memory LSTM)
# RNN Classifier
# # keras.layers import SimpleRNN Dense, Activation, Convolution2D, MaxPooling2D, Flatten

# Autoencoder
# (x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
#
# # data pre-processing
# x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
# x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
# x_train = x_train.reshape((x_train.shape[0], -1))
# x_test = x_test.reshape((x_test.shape[0], -1))
# # print(x_train.shape)
# # print(x_test.shape)
# # this is our input placeholder
# input_img = keras.layers.Input(shape=(784,))
# encoding_dim = 2
#
# # encoder layers
# encoded = keras.layers.Dense(128, activation='relu')(input_img)
# encoded = keras.layers.Dense(64, activation='relu')(encoded)
# encoded = keras.layers.Dense(10, activation='relu')(encoded)
# encoder_output = keras.layers.Dense(encoding_dim)(encoded)
#
# # decoder layers
# decoded = keras.layers.Dense(10, activation='relu')(encoder_output)
# decoded = keras.layers.Dense(64, activation='relu')(decoded)
# decoded = keras.layers.Dense(128, activation='relu')(decoded)
# decoded = keras.layers.Dense(784, activation='tanh')(decoded)
#
# # construct the autoencoder model
# autoencoder = keras.models.Model(inputs=input_img, outputs=decoded)
#
# # construct the encoder model for plotting
# encoder = keras.models.Model(inputs=input_img, outputs=encoder_output)
#
# autoencoder.compile(optimizer='adam', loss='mse')
#
# autoencoder.fit(x_train, x_train,
#                 epochs=20,
#                 batch_size=256,
#                 shuffle=True)
# encoded_imgs = encoder.predict(x_test)
# plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
# plt.colorbar()
# plt.show()


















































































































