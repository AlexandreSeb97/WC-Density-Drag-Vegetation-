# Original MATLAB Water Column Master Code from Lisa Lucas, modified by Tina Chow. Spring 2018: Mark Stacey and Michaella Chung.
# Code adapted to Python and for USGS by Alexandre E. S. Georges, Environmental Engineering PhD Student at UC Berkeley - 2022
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.animation as animation

from numba import jit
from column import Column
from params import *
from advance import wc_advance


# Test

# Column initialization
# Column(N, H, Length, SMALL)
c1 = Column(80, 20, 10, SMALL) # Creation of column object
c1.setup(A, B, C, Sq, kappa, SMALL, nu, g, rho0, alpha) # Filling in parameter values
t = [] 

fig, ax = plt.subplots()
ax.autoscale
line, = ax.plot([], [], lw = 3)

def init():
    line.set_data([], [])
    return line,

@jit
def animate(i):
    t.append(c1.dt*(i))
    ax.clear()
    # Unew, Cnew, Qnew, Q2new, Q2Lnew, rhonew, Lnew, nu_tnew, Kznew, Kqnew, N_BVnew, N_BVsqnew
    [c1.U, c1.scalar, c1.Q, c1.Q2, c1.Q2L, c1.rho, c1.L, c1.nu_t, c1.Kz, c1.Kq, c1.N_BV, c1.N_BVsq] = wc_advance(c1, t_px, px0, t[i])
    print('Step! t=' +str(t[i])+'s')
    ax.plot(c1.scalar, c1.z)

    ax.grid()


ani = FuncAnimation(fig, animate, frames = c1.M, interval=c1.dt, repeat=False)
#ani.save('Animations/test.gif', writer='imagemagick', fps=24)
plt.show()
