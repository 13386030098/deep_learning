import tensorflow as tf
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


#tf Variable
# state = tf.Variable(0, name='counter')
# tf placeholder
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
#
# output = tf.multiply(input1,input2)
#
# with tf.Session() as sess:
#     print(sess.run(output, feed_dict={input1:[7.], input2:[2.]}))

#Activation Function
#first tensorflow network

#
# def add_layer(inputs, in_size, out_size, activation_function=None):
#     Weights = tf.Variable(tf.random_normal([in_size, out_size]))
#     biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
#     Wx_plus_b = tf.matmul(inputs, Weights) + biases
#     if activation_function is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_function(Wx_plus_b)
#     return outputs
#
# x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
# y_data = np.square(x_data) - 0.5 + noise
#
# with tf.name_scope('inputs'):
#     xs = tf.placeholder(tf.float32, [None, 1], name= 'x_input')
#     ys = tf.placeholder(tf.float32, [None, 1], name= 'y_input')
#
# l1 = add_layer(x_data, 1, 10, activation_function=tf.nn.relu)
# prediction = add_layer(l1, 10, 1, activation_function=None)
#
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),
#                      reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()#本次运行请注释，全局运行不要注释
# plt.show()
#
#
# for i in range(10000):
#     sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
#     if i % 50 ==0:
#         try:
#             ax.lines.remove(lines[0])
#         except Exception:
#             pass
#         prediction_value = sess.run(prediction, feed_dict={xs: x_data})
#         # plot the prediction
#         lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
#         plt.pause(0.1)
#         print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))

#Speed Up Training
#Optimizer
#Tensorboard

# def add_layer(inputs, in_size, out_size, activation_function=None):
#     layer_name = 'layer' #n_layer
#     with tf.name_scope(layer_name):
#         with tf.name_scope('weights'):
#             Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
#             tf.summary.histogram(layer_name + '/weights', Weights)
#         with tf.name_scope('biases'):
#             biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
#             tf.summary.histogram(layer_name + '/biases', biases)
#         with tf.name_scope('Wx_plus_b'):
#             Wx_plus_b = tf.matmul(inputs, Weights) + biases
#         if activation_function is None:
#             outputs = Wx_plus_b
#         else:
#             outputs = activation_function(Wx_plus_b)
#         tf.summary.histogram(layer_name + '/outputs', outputs)
#     return outputs
#
#
# x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
# noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
# y_data = np.square(x_data) - 0.5 + noise
#
#
# with tf.name_scope('inputs'):
#     xs = tf.placeholder(tf.float32, [None, 1], name= 'x_input')
#     ys = tf.placeholder(tf.float32, [None, 1], name= 'y_input')
#
# # add hidden layer
# l1 = add_layer(x_data, 1, 10, activation_function=tf.nn.relu)
# # add output layer
# prediction = add_layer(l1, 10, 1, activation_function=None)
#
# with tf.name_scope('loss'):
#     loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),
#                      reduction_indices=[1]))
#     tf.summary.scalar('loss', loss)
# with tf.name_scope('train'):
#     train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
#
# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter("logs/", sess.graph)
#
# sess.run(init)
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()#本次运行请注释，全局运行不要注释
# plt.show()
#
#
# for i in range(1000):
#     sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
#     if i % 10 ==0:
#         try:
#             ax.lines.remove(lines[0])
#         except Exception:
#             pass
#         prediction_value = sess.run(prediction, feed_dict={xs: x_data})
#         # plot the prediction
#         lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
#         plt.pause(0.1)
#         print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
#         result = sess.run(merged,
#                           feed_dict={xs: x_data, ys: y_data})
#         writer.add_summary(result, i)


#Overfitting method: Dropout
# load data
# digits = load_digits()
# X = digits.data
# y = digits.target
# y = LabelBinarizer().fit_transform(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
#
# def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
#     # add one more layer and return the output of this layer
#     Weights = tf.Variable(tf.random_normal([in_size, out_size]))
#     biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
#     Wx_plus_b = tf.matmul(inputs, Weights) + biases
#     # here to dropout
#     Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
#     if activation_function is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_function(Wx_plus_b, )
#     tf.summary.histogram(layer_name + '/outputs', outputs)
#     return outputs
#
#
# # define placeholder for inputs to network
# keep_prob = tf.placeholder(tf.float32)
# xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
# ys = tf.placeholder(tf.float32, [None, 10])
#
# # add output layer
# l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
# prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)
#
# # the loss between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1]))  # loss
# tf.summary.scalar('loss', cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# sess = tf.Session()
# merged = tf.summary.merge_all()
# # summary writer goes in here
# train_writer = tf.summary.FileWriter("logs/train", sess.graph)
# test_writer = tf.summary.FileWriter("logs/test", sess.graph)
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# for i in range(500):
# # here to determine the keeping probability
#     sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
#     if i % 50 == 0:
#         # record loss
#         train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
#         test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
#         train_writer.add_summary(train_result, i)
#         test_writer.add_summary(test_result, i)

#Convolutional Neural Network (CNN)
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)
#
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
#
# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
#     correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
#     return result
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
# def conv2d(x, W):
#     # stride [1, x_movement, y_movement, 1]
#     # Must have strides[0] = strides[3] = 1
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
# def max_pool_2x2(x):
#     # stride [1, x_movement, y_movement, 1]
#     return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#
# # define placeholder for inputs to network
# xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
# ys = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32)
# x_image = tf.reshape(xs, [-1, 28, 28, 1])
# # print(x_image.shape)  # [n_samples, 28,28,1]
#
# ## conv1 layer ##
# W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
# b_conv1 = bias_variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
# h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32
#
# ## conv2 layer ##
# W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
# h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64
#
# ## fc1 layer ##
# W_fc1 = weight_variable([7*7*64, 1024])
# b_fc1 = bias_variable([1024])
# # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# ## fc2 layer ##
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_variable([10])
# prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
#
# # the error between prediction and real data
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
#                                               reduction_indices=[1]))       # loss
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
# sess = tf.Session()
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
#     if i % 50 == 0:
#         print(compute_accuracy(
#             mnist.test.images[:1000], mnist.test.labels[:1000]))

#Autoencoder










































































































































































