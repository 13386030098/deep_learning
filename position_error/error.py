import matplotlib.pyplot as plt
import  numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import animation
from matplotlib import cm


fig=plt.figure(dpi = 200)
ax=Axes3D(fig)


a = np.loadtxt('plan_x.txt')
b = np.loadtxt('plan_y.txt')
c = np.loadtxt('plan_z.txt')

# ax.plot(a, b, 'r+', zdir='c', zs=0.44)


# x = np.loadtxt('real_x.txt')
# y = np.loadtxt('real_y.txt')
# z = np.loadtxt('real_z.txt')
e = np.loadtxt('error.txt')

x=a-e*4
y=b-e*3
z=c
# # # ax.plot(x, y, 'g+', zdir='z', zs=0.44)
# e = np.loadtxt('error.txt')
# e = np.loadtxt('error_compa.txt')



p1=ax.scatter(a, b, c, c='r')
p2=ax.scatter(x, y, z, c='g')







# colors = cm.hsv((e)/max(e))
# colmap = cm.ScalarMappable(cmap=cm.hsv)
# colmap.set_array(e)
# yg = ax.scatter(a, b, c, c=colors, marker='o')
# cb = fig.colorbar(colmap)






plt.legend([p1, p2], ['set point', 'actual point'], loc='best', scatterpoints=1)

ax.set_zlabel('Z(m)')  # 坐标轴
ax.set_ylabel('Y(m)')
ax.set_xlabel('X(m)')

plt.show()
































