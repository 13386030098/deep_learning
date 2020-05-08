import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.autograd import  Variable
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import  torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import copy

#文件
# file = open('test.txt','w')
# file.write("hello world")
# file.close()


# file = open('test.txt','r')
# content = file.read()#全部读取
# content = file.readline()

#列表
# list = [1,2,3,4]
# for i in list:
#     print(i)

# for index in range(len(list)):
#     print('index=',index,'number=',list[index])

# list = [1,2,3,4]
# list.append(5)
# print(list)

# list.insert(0,0)#序号＋数据
# print(list)

# print(list[:3])#前三位
# print(list[-3:])
# print(list.count(1))
# list.sort(reverse = True)
# print(list)


# list = [[1,2,3],[4,5,6],[7,8,9]]
# print(list)

# try:
#     file = open('2.txt','w')
# except Exception as e:
#     print(e)
#
# else:
#     file.write('')
#     file.close()

#
# a = [1,2,3]
# b = [4,5,6]
# print(list(zip(a,b)))

# for i,j in zip(a,b):
#     print(i,j)

# a = [1,2,3]
# b=a
# print(id(a))
# print(id(b))
# b[0]=5
# print(a)

# b=copy.copy(a)#浅复制
# print(id(a)==id(b))

# b=copy.deepcopy(a)

#多线程









#多进程

































