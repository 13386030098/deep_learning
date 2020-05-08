import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math
import time


np.random.seed(1)

data = np.loadtxt('data.txt')


#
# col_1 = data[:,0]
# col_2 = data[:,1]
# col_3 = data[:,2]
#
# col_4 = data[:,3]
# col_5 = data[:,4]
# col_6 = data[:,5]
#
# col_7 = data[:,6]
# col_8 = data[:,7]
# col_9 = data[:,8]
#
#
# data_plot_1 = col_1[10000:11000]
# data_plot_2 = col_2[10000:11000]
# data_plot_3 = col_3[10000:11000]
#
# data_plot_4 = col_4[10000:11000]
# data_plot_5 = col_5[10000:11000]
# data_plot_6 = col_6[10000:11000]
#
# data_plot_7 = col_7[10000:14000]
# data_plot_8 = col_8[10000:14000]
# data_plot_9 = col_9[10000:14000]
#
# # plt.plot(data_plot_7)
# # plt.show()
#
# data_plot_1_filter = scipy.signal.savgol_filter(col_1, 51, 10)
# data_plot_2_filter = scipy.signal.savgol_filter(col_2, 51, 10)
# data_plot_3_filter = scipy.signal.savgol_filter(col_3, 51, 10)
#
# data_plot_4_filter = scipy.signal.savgol_filter(col_4, 51, 10)
# data_plot_5_filter = scipy.signal.savgol_filter(col_5, 51, 10)
# data_plot_6_filter = scipy.signal.savgol_filter(col_6, 51, 10)
#
# data_plot_7_filter = scipy.signal.savgol_filter(col_7, 51, 10)
# data_plot_8_filter = scipy.signal.savgol_filter(col_8, 51, 10)
# data_plot_9_filter = scipy.signal.savgol_filter(col_9, 51, 10)
#
# yhat_acc_1 = scipy.signal.savgol_filter(data_plot_4_filter, 51, 8, 1, 0.02) # window size 51, polynomial order 3
# yhat_acc_2 = scipy.signal.savgol_filter(data_plot_5_filter, 51, 8, 1, 0.02) # window size 51, polynomial order 3
# yhat_acc_3 = scipy.signal.savgol_filter(data_plot_6_filter, 51, 8, 1, 0.02) # window size 51, polynomial order 3



data = np.loadtxt('torque_collision_error.txt')

col_1 = data[160:1900]
# col_2 = data[104:,1]
# col_3 = data[104:,2]


plt.figure()
# plt.subplot(1,3,1)
l1,= plt.plot(col_1, label='roll_torque_collision_error')
plt.legend(handles=[l1],labels=['roll_torque_collision_error'],loc='best')
plt.ylabel('F(Nm)')
plt.xlabel('t')

# plt.figure()
# # plt.subplot(1,3,2)
# l1,= plt.plot(col_2, label='link_torque_error')
# plt.legend(handles=[l1],labels=['link_torque_error'],loc='best')
# plt.ylabel('F(Nm)')
# plt.xlabel('t')
#
# plt.figure()
# # plt.subplot(1,3,3)
# l1,= plt.plot(col_3, label='slide_torque_error')
# plt.legend(handles=[l1],labels=['slide_torque_error'],loc='best')
# plt.ylabel('F(Nm)')
# plt.xlabel('t')
plt.show()
#



# Q = 1e-3 # process variance
# R = 0.1**2 # estimate of measurement variance, change to see effect

# z = col_7[:2000]
#
# n_iter = col_7[:2000].shape[0]
# sz = (n_iter,) # size of array


# xhat = np.zeros(sz)      # a posteri estimate of x 滤波估计值
# P = np.zeros(sz)         # a posteri error estimate滤波估计协方差矩阵
# xhatminus = np.zeros(sz) # a priori estimate of x 估计值
# Pminus = np.zeros(sz)    # a priori error estimate估计协方差矩阵
# K = np.zeros(sz)         # gain or blending factor卡尔曼增益
#
# xhat[0] = col_7[0]
# P[0] = 1.0

# for k in range(1,n_iter):
#     # 预测
#     xhatminus[k] = xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
#     Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
#
#     # 更新
#     K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
#     xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
#     P[k] = (1 - K[k]) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1



# for position, velocity
#
# Q = 1e-3 # process variance
# R = 0.01**2 # estimate of measurement variance, change to see effect
#
# # torque
# # Q = 1e-3 # process variance
# # R = 0.1**2 # estimate of measurement variance, change to see effect
#
# test_array = np.zeros(col_4.shape[0])
# n_iter = col_4.shape[0]
# xhat = 0
# z = col_4
#
# xhatminus = 0 # a priori estimate of x 估计值
# Pminus = 0    # a priori error estimate估计协方差矩阵
# K = 0         # gain or blending factor卡尔曼增益
# P = 1.0
#
# for k in range(n_iter):
#     # 预测
#     xhatminus = xhat  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
#     Pminus = P + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
#
#     # 更新
#     K = Pminus / (Pminus + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
#     xhat = xhatminus + K * (z[k] - xhatminus)  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
#     P = (1 - K) * Pminus  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
#     test_array[k] = xhat
#     # test_acc[k] = (test_array[k] - test_array[k -1])/0.02
#
# np.savetxt('tor_dim_3.txt', test_array)
# a = np.loadtxt('tor_dim_3.txt')
# a = np.transpose(a)
# print(a.shape)
#
# l1,= plt.plot(data_plot_4_filter[:2000])
# l2,= plt.plot(a[:2000])
# plt.legend(handles=[l1, l2], labels=['S-G', 'kalman'],loc='best')
# plt.xlabel('t')
# plt.ylabel('F(Nm)')
# plt.show()



## position and velocity to calculate acceleration

# z1 = data_plot_1_filter.reshape(1,60000)
# z2 = data_plot_4_filter.reshape(1,60000)

#
# z1 = np.loadtxt('data_ver_3/pos_dim_3.txt').reshape(1,60000)
# z2 = np.loadtxt('data_ver_3/vel_dim_3.txt').reshape(1,60000)
#
# z = np.concatenate((z1, z2), 0)
#
# z = np.mat(z)
# # print(type(z))
#
# acc = 0.5
#
# delta_t = 0.02
#
# A = np.mat([[1 , delta_t, 1/2*delta_t**2], [0, 1, delta_t], [0, 0, 1]])
# H = np.mat([[1, 0, 0], [0, 1, 0]])
#
#
# P = np.mat([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])# error covarianc 3*3
# Q = np.mat([[0.5e-3, 0, 0], [0, 0.5e-3, 0], [0, 0, 2e-1]]) # process variance 3*3
# R = np.mat([[0.0001**2, 0], [0, 0.0001**2]]) #measurement variance 2*2
#
# xhat = np.mat(np.zeros((3,1)))
#
# # xhat[0] = z1[0,0]
# # xhat[1] = z2[0,0]
# # xhat[2] = acc
#
# xhatminus = np.mat(np.zeros((3,1)))
# Pminus = np.mat(np.zeros((3,3)))
#
# K= np.mat(np.zeros((3,2)))
# I = np.mat(np.identity(3))
#
# test_array = np.zeros(col_１.shape[0])
# n_iter = col_１.shape[0]
#
# for k in range(n_iter):
#     xhatminus = A*xhat
#     Pminus = A*P*A.T + Q
#
#     K = Pminus * H.T* np.linalg.inv(H*Pminus*H.T + R)
#     xhat = xhatminus + K*(z[:,k] - H*xhatminus)
#     P = (I - K*H) * Pminus
#     test_array[k] = xhat[2]
#
# np.savetxt('acc_dim_3.txt', test_array)
# a = np.loadtxt('acc_dim_3.txt')
# a = np.transpose(a)
# print(a.shape)
#
# l1,= plt.plot(test_array[:2000])
# l2,= plt.plot(yhat_acc_3[:2000])
# plt.legend(handles=[l1, l2], labels=['kalman_acc', 'S-G'],loc='best')
# plt.show()












# plt.plot(data_plot_8)
# plt.plot(data_plot_8_filter[10000:14000])
# plt.show()

# l1,= plt.plot(data_plot_7)
# l2,= plt.plot(data_plot_8)
# l3,= plt.plot(data_plot_9)
#
# l4,= plt.plot(data_plot_7_filter[10000:14000])
# l5,= plt.plot(data_plot_8_filter[10000:14000])
# l6,= plt.plot(data_plot_9_filter[10000:14000])

# plt.legend(handles=[l1,l2,l3,l4,l5,l6],labels=['roll','link','instrument','roll_filter','link_filter','instrument_filter'],loc='best')
#
# plt.xlabel('t')
# plt.ylabel('F(Nm)')
# plt.show()

# np.savetxt('force.txt',(data_plot_7_filter,data_plot_8_filter,data_plot_9_filter))
# a = np.loadtxt('force.txt')
# a = np.transpose(a)
# print(a.shape)


# yhat_acc_1 = scipy.signal.savgol_filter(data_plot_4_filter, 51, 8, 1) # window size 51, polynomial order 3
# yhat_acc_2 = scipy.signal.savgol_filter(data_plot_5_filter, 51, 8, 1) # window size 51, polynomial order 3
# yhat_acc_3 = scipy.signal.savgol_filter(data_plot_6_filter, 51, 8, 1) # window size 51, polynomial order 3


# l2,= plt.plot(yhat_acc_1[10000:14000])
# plt.show()

# np.savetxt('acc.txt',(yhat_acc_1,yhat_acc_2,yhat_acc_3))
# a = np.loadtxt('acc.txt')
# a = np.transpose(a)
# print(a.shape)


# np.savetxt('pos_vel.txt',(data_plot_1_filter,data_plot_2_filter,data_plot_3_filter,data_plot_4_filter,data_plot_5_filter,data_plot_6_filter))
# a = np.loadtxt('pos_vel.txt')
# a = np.transpose(a)
# print(a.shape)

