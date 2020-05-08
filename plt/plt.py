
import matplotlib.pyplot as plt
import  numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib import animation

# x=np.linspace(-3,3,50)
# plt.figure()
#  y1=2*x+1
#  plt.plot(x,y1)

# plt.figure(figsize=(8, 5))
# y1=2*x+1
# y2=x**2
# l1,=plt.plot(x,y1,label='up')
# l2,=plt.plot(x,y2,color='red',linewidth=2.0,linestyle='-',label='down')
# plt.legend(handles=[l1,l2],labels=['up','down'],loc='best')
#
# #添加注释
# x0=0
# y0=2*x0+1
# plt.scatter(x0,y0,s=50,color='b')
# plt.plot([x0,x0],[y0,0],'k--',lw=2.5)
#
# plt.annotate(r'$2x+1=%s$'% y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),
#              textcoords='offset points', fontsize=16,
#              arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
# plt.text(0, 0, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$')

# plt.text(0, 0, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
#          fontdict={'size': 16, 'color': 'r'})

#tick能见度
#散点图
# n=1024
# x=np.random.normal(0,1,n)
# y=np.random.normal(0,1,n)
# T=np.arctan2(y,x)
# plt.scatter(x,y,s=75,c=T,alpha=0.5)
# plt.scatter(np.arange(5),np.arange(5))
# plt.xlim((-1.5,1.5))
# plt.ylim((-1.5,1.5))


#柱状图
# n=12
# X=np.arange(n)
# y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
# y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
#
# plt.bar(X,+y1,facecolor='#9999ff',edgecolor='white')
# plt.bar(X,-y2,facecolor='#ff9999',edgecolor='white')
#
# for x, y in zip(X, y1):
#     # ha: horizontal alignment
#     # va: vertical alignment
#     plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
#
# for x, y in zip(X,y2):
#     # ha: horizontal alignment
#     # va: vertical alignment
#     plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')

#3D数据
# fig=plt.figure()
# ax=Axes3D(fig)
# # X, Y value
# X = np.arange(-4, 4, 0.25)
# Y = np.arange(-4, 4, 0.25)
# X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
# R=np.sqrt(X**2+Y**2)
# Z=np.sin(R)
# ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
# ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
# ax.set_zlim(-2,2)

#多图合并显示
#plt.figure()
# plt.subplot(2,2,1)
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,2,2)
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,2,3)
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,2,4)
# plt.plot([0,1],[0,1])


# plt.subplot(2,1,1)
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,3,4)
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,3,5)
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,3,6)
# plt.plot([0,1],[0,1])


#method3
# ax1=plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)
# ax1.plot([1,2],[1,2])
# ax1.set_title('ax1_title')
# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax5 = plt.subplot2grid((3, 3), (2, 1))

# gs=gridspec.GridSpec(3,3)
# ax6 = plt.subplot(gs[0, :])
# ax7 = plt.subplot(gs[1, :2])
# ax8 = plt.subplot(gs[1:, 2])
# ax9 = plt.subplot(gs[-1, 0])
# ax10 = plt.subplot(gs[-1, -2])

# f,((ax11,ax12),(ax21,ax22))=plt.subplots(2,2,sharex=True, sharey=True)
# ax11.scatter([1,2],[1,2])

#次坐标轴
















# plt.xlim(-1,2)
# plt.ylim(-2,3)
# plt.xlabel('x')
# plt.ylabel('y')

# new_ticks=np.linspace(-1,2,5)
# plt.xticks(new_ticks)
# plt.yticks([-2,-1.8,-1,1.22,3,],[r'$really\ good$',r'$nn\ \alpha$','hjc','hjbcas',])

#gca='get current axis'
# ax=plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data',-1))
# ax.spines['left'].set_position(('data',0))

#绘制图例

plt.show()
















































