"""
Run the pikaia genetic algorithm on a 2D problem, plot the objective
function, the convergence of the run and show an animation of the run, where
(all) individuals should converge to the optimum.

The objective function `ff` is not the "twod" problem from the pikaia source,
but rather a x^2-like landscape with many local maxima.

Notes:
* In contrast to most optimization examples, which are minimizations,
  the pikaia code does a maximization, i.e. we need to make sure that the
  optimum has the highest function value.
* The parameter values must be in [0,1], so the function to be optimized must
  scale it's variables accordingly.
* The more individuals, the better the converge to the true optimum.  
"""

import numpy as np
from scipy.optimize import fmin
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d 
import matplotlib.animation as animation
from ffnet import pikaia

sin = np.sin
cos = np.cos

# Define objective function on x-y space x,y=-k...k with v=[x,y]
k = 20.0
ff = lambda v: -(((-k+v[0]*2*k)**2.0 + (-k+v[1]*2*k)**2.0 )/70.0 + sin(-k+v[0]*2*k) + sin(-k+v[1]*2*k))

# optimum: [ 0.4622,  0.4622]
xtrue = fmin(lambda v: -ff(v), [.45,.45])
ytrue = ff(xtrue)

# for plotting ff
#
# Shameless plug: Get rid of all the meshgrid madness? See pwtools.mpl.Data2D,
# https://bitbucket.org/elcorto/pwtools
x = np.linspace(0,1,50)
nx = len(x); ny=nx
X,Y = np.meshgrid(x,x)
X = X.T; Y = Y.T
xx = X.flatten()
yy = Y.flatten()
XY = np.array([xx, yy]).T
Z = np.array([ff(v) for v in XY]).reshape(nx, ny)

# run genetic algorithm
xopt = pikaia(ff, 2, individuals=80, generations=30, maxrate=0.8, 
              reproduction=3, verbosity=3)


#----------------------------------------------------------------------------
# Plot objective function and convergence of best,mean,worst individual.
#----------------------------------------------------------------------------

d = np.loadtxt('pikaia_fit.txt') 
fig0 = plt.figure()
ax0 = fig0.add_subplot(111)
ax0.plot(d[:,1:4])
ax0.legend(('best', 'mean', 'worst'), loc='lower right')
xlim = ax0.get_xlim()
ax0.hlines(ytrue, xlim[0], xlim[1], color='k', linestyle='--')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z, cmap=cm.jet, rstride=1, cstride=1) 
ax1.plot([xopt[0]], [xopt[1]], [ff(xopt)], 'mo', ms=10)

print("pikaia solution: {}".format(xopt))
print("real   solution: {}".format(xtrue))


#----------------------------------------------------------------------------
# Convergence animation
#----------------------------------------------------------------------------

# format of pikaia_ind_all.txt: ii i1_x i1_y i2_x i2_y ... where ii =
# generation and i1,i2,... are the individuals 1,2,...,np (the points in x-y
# space, np=number of individuals in pikaia.f).
d = np.loadtxt('pikaia_ind_all.txt')

fig2,ax2 = plt.subplots()
def update_ax(ii):
    ax2.cla() # needed!
    ax2.contourf(X, Y, Z, 70)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_title("iter = {:5}".format(ii))
    ax2.plot(xtrue[0], xtrue[1], 'mo', ms=30, fillstyle='none', markeredgewidth=4)
    ax2.scatter(d[ii,1::2], d[ii,2::2], s=30)

ani = animation.FuncAnimation(fig2, update_ax, frames=d.shape[0],
                              blit=False, interval=100)


plt.show()

