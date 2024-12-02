import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation

W = 100
a = 2
n = 2
delta1 = 0.1
delta2 = 0.2

b = np.linspace(0, 5, num=51)
d = np.linspace(0, 5, num=51)

B, D = np.meshgrid(b, d)

mmp = (6.895 * W)/(n * a * pow(B,0.8) * pow(D,0.8) * pow(delta2,0.4))

tri = Triangulation(B.ravel(), D.ravel())
 
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
 
ax.plot_trisurf(tri, mmp.ravel(), cmap='cool', edgecolor='none', alpha=0.8)

ax.set_title('MMP metric as a function of d and  (delta=0.2)', fontsize=14)
ax.set_xlabel('b', fontsize=12)
ax.set_ylabel('d', fontsize=12)
ax.set_zlabel('MMP', fontsize=12)

plt.show()