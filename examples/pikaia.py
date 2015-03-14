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
import matplotlib.animation as animation
import os
from ffnet import pikaia
from pwtools import mpl

sin = np.sin
cos = np.cos
plt = mpl.plt

# Define objective function on x-y space x,y=-k...k with v=[x,y]
k = 20.0
ff = lambda v: -(((-k+v[0]*2*k)**2.0 + (-k+v[1]*2*k)**2.0 )/70.0 + sin(-k+v[0]*2*k) + sin(-k+v[1]*2*k))

# optimum: array([ 0.4622,  0.4622])
xm = fmin(lambda v: -ff(v), [.45,.45])
ym = ff(xm)

# for plotting ff
x = np.linspace(0,1,50) 
dd = mpl.Data2D(x=x,y=x) 
dd.update(zz=np.array([ff(v) for v in dd.XY]))

xopt = pikaia(ff, 2, individuals=80, generations=30, maxrate=0.8, 
              reproduction=3, verbosity=3)


#----------------------------------------------------------------------------
# Plot objective function and convergence of best,mean,worst individual.
#----------------------------------------------------------------------------

d = np.loadtxt('pikaia_fit.txt') 
fig0,ax0 = plt.subplots()
ax0.plot(d[:,1:4])
ax0.legend(('best', 'mean', 'worst'), loc='lower right')
xlim = ax0.get_xlim()
ax0.hlines(ym, xlim[0], xlim[1], color='k', linestyle='--')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(dd.X, dd.Y, dd.Z, cmap=cm.jet, rstride=1, cstride=1) 
ax1.plot([xopt[0]], [xopt[1]], [ff(xopt)], 'mo', ms=10)

print("pikaia solution: {}".format(xopt))
print("real   solution: {}".format(xm))


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
    ax2.contourf(dd.X, dd.Y, dd.Z, 70)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_title("iter = {:5}".format(ii))
    ax2.plot(xm[0], xm[1], 'mo', ms=30, fillstyle='none', markeredgewidth=4)
    ax2.scatter(d[ii,1::2], d[ii,2::2], s=30)

ani = animation.FuncAnimation(fig2, update_ax, frames=d.shape[0],
                              blit=False, interval=100)
plt.show()

