import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error

import scipy.io
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
np.random.seed(1)


pos_dim_1 = np.loadtxt('data_ver_3/pos_dim_1.txt').astype(np.float32)
pos_dim_2 = np.loadtxt('data_ver_3/pos_dim_2.txt').astype(np.float32)
pos_dim_3 = np.loadtxt('data_ver_3/pos_dim_3.txt').astype(np.float32)

vel_dim_1 = np.loadtxt('data_ver_3/vel_dim_1.txt').astype(np.float32)
vel_dim_2 = np.loadtxt('data_ver_3/vel_dim_2.txt').astype(np.float32)
vel_dim_3 = np.loadtxt('data_ver_3/vel_dim_3.txt').astype(np.float32)

acc_dim_1 = np.loadtxt('data_ver_3/acc_dim_1.txt').astype(np.float32)
acc_dim_2 = np.loadtxt('data_ver_3/acc_dim_2.txt').astype(np.float32)
acc_dim_3 = np.loadtxt('data_ver_3/acc_dim_3.txt').astype(np.float32)

tor_dim_1 = np.loadtxt('data_ver_3/tor_dim_1.txt').astype(np.float32)
tor_dim_2 = np.loadtxt('data_ver_3/tor_dim_2.txt').astype(np.float32)
tor_dim_3 = np.loadtxt('data_ver_3/tor_dim_3.txt').astype(np.float32)


pos_dim_1 = pos_dim_1.reshape(-1, 1)
pos_dim_2 = pos_dim_2.reshape(-1, 1)
pos_dim_3 = pos_dim_3.reshape(-1, 1)

vel_dim_1 = vel_dim_1.reshape(-1, 1)
vel_dim_2 = vel_dim_2.reshape(-1, 1)
vel_dim_3 = vel_dim_3.reshape(-1, 1)

acc_dim_1 = acc_dim_1.reshape(-1, 1)
acc_dim_2 = acc_dim_2.reshape(-1, 1)
acc_dim_3 = acc_dim_3.reshape(-1, 1)

tor_dim_1 = tor_dim_1.reshape(-1, 1)
tor_dim_2 = tor_dim_2.reshape(-1, 1)
tor_dim_3 = tor_dim_3.reshape(-1, 1)


assert(pos_dim_1.shape == (60000, 1))
assert(pos_dim_2.shape == (60000, 1))
assert(pos_dim_3.shape == (60000, 1))

assert(vel_dim_1.shape == (60000, 1))
assert(vel_dim_2.shape == (60000, 1))
assert(vel_dim_3.shape == (60000, 1))

assert(acc_dim_1.shape == (60000, 1))
assert(acc_dim_2.shape == (60000, 1))
assert(acc_dim_3.shape == (60000, 1))

assert(tor_dim_1.shape == (60000, 1))
assert(tor_dim_2.shape == (60000, 1))
assert(tor_dim_3.shape == (60000, 1))


input_data  = np.concatenate((pos_dim_1, pos_dim_2, pos_dim_3, vel_dim_1, vel_dim_2, vel_dim_3,
                              acc_dim_1, acc_dim_2, acc_dim_3), 1)
output_date = np.concatenate((tor_dim_1, tor_dim_2, tor_dim_3), 1)

assert(input_data.shape == (60000, 9))
assert(output_date.shape == (60000, 3))


X_train, X_val, X_test = input_data[:56000,:], input_data[56000:58000,:], input_data[58000:60000,:]
Y_train, Y_val, Y_test = output_date[:56000,:], output_date[56000:58000,:], output_date[58000:60000,:]


# X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
# X_val = (X_val - np.mean(X_val, axis=0)) / np.std(X_val, axis=0)
# X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# input_dim = 9
# output_dim = 3
# #
# def build_model():
#     model = tf.keras.Sequential([
#         layers.Dense(256, activation='relu',
#                      input_shape=(input_dim,)),
#         layers.Dropout(0.4),
#         layers.Dense(256, activation='relu'),
#         layers.Dropout(0.4),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(0.4),
#         layers.Dense(output_dim)
#     ])
#
#     # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     #     0.001,
#     #     decay_steps=STEPS_PER_EPOCH * 1000,
#     #     decay_rate=1,
#     #     staircase=False)
#
#     global_step = tf.Variable(0)
#     learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.9, staircase=True)
#
#     optimizer = keras.optimizers.Adam(learning_rate)
#
#     model.compile(loss='mse',
#                   optimizer=optimizer,
#                   metrics=['mae', 'mse'])
#     return model
#
# model = build_model()
# # # #
# model.summary()
#
# tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
#                                          histogram_freq=1,
#                                          batch_size=2000,
#                                          write_graph=True,
#                                          write_images=True,
#                                          )
# model.fit(X_train, Y_train,
#           epochs=2000,
#           batch_size=128,
#           validation_data=(X_val, Y_val),
#           callbacks=[tbCallBack]
#           )
#
# model.save_weights('./checkpoints/my_checkpoint')
# model.load_weights('./checkpoints/my_checkpoint')

# model.save('predict_model.h5')
model = keras.models.load_model('predict_model.h5')

# predictions = model.predict(X_test)
# pred_train = model.predict(X_train)
print(X_test[0])
pred_train = model.predict(X_test[0].reshape(1,9))
print(pred_train)
# pred_test = model.predict(X_test)
start = time.time()
predictions = model.predict(X_test[0].reshape(1,9))
stop = time.time()
print(str((stop-start)*1000) + "ms")


# loss = model.evaluate(X_test, Y_test, batch_size=128)
# print('test loss: ', loss)
# # stop = time.time()
# # print(str((stop-start)*1000) + "ms")
# ## The nMSE is the mean squared error
# ## divided by the variance of the corresponding output dimension
# F= Y_train.shape[1]
# var = np.var(Y_train, axis=0)
# for f in range(F):
#     # nMSE_train = mean_squared_error(Y_train[:, f] , pred_train[:, f])/var[f]
#     # nMSE_test = mean_squared_error(Y_test[:, f] , pred_test[:, f])/var[f]
#     nMSE_train = sum((Y_train[:, f] - pred_train[:, f])**2) / len(Y_train) / var[f]
#     nMSE_test = sum((Y_test[:, f] - pred_test[:, f])**2) / len(Y_test) / var[f]
#     print("Dimension %d: nMSE = %f%% (training) / %f%% (validation)"
#           % (f + 1, nMSE_train * 100, nMSE_test * 100))
# #
# plt.figure()
# # plt.subplot(1,3,1)
# l1,= plt.plot(Y_test[:2000,0], label='test_data')
# l2,= plt.plot(predictions[:2000,0], label='predict_data')
# plt.legend(handles=[l1,l2],labels=['roll_test_data','roll_predict_data'],loc='best')
# plt.ylabel('F(Nm)')
# plt.xlabel('t')
#
# plt.figure()
# # plt.subplot(1,3,2)
# l1,= plt.plot(Y_test[:2000,1], label='test_data')
# l2,= plt.plot(predictions[:2000,1], label='predict_data')
# plt.legend(handles=[l1,l2],labels=['link_test_data','link_predict_data'],loc='best')
# plt.ylabel('F(Nm)')
# plt.xlabel('t')
#
# plt.figure()
# # plt.subplot(1,3,3)
# l1,= plt.plot(Y_test[:2000,2], label='test_data')
# l2,= plt.plot(predictions[:2000,2], label='predict_data')
# plt.legend(handles=[l1,l2],labels=['slide_test_data','slide_predict_data'],loc='best')
# plt.ylabel('F(Nm)')
# plt.xlabel('t')
# plt.show()



























































