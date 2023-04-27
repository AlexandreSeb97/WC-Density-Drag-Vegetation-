# Object for Water Column. Parameters are:
# U, V, Q2, Q2L, Q2L, L, temp, Gh, Sm, Sh, Kq, nu_t, Kz

import math
import numpy as np
import pandas as pd

class Column:
    def __init__(self, N, H, Len, SMALL):
        self.z = np.zeros(N) ##### In the future separate N in Nx and Nz
        self.x = np.zeros(N) ##### In the future separate N in Nx and Nz
        self.U = np.zeros(N)
        self.V = np.zeros(N)
        self.N_BV = np.zeros(N)
        self.N_BVsq = np.zeros(N)
        self.rho = np.zeros(N)
        self.Q2 = np.ones(N)*SMALL # Turbulent field is seeded with small values then let evolve
        self.Q2L = np.ones(N)*SMALL 
        self.Q = np.zeros(N)
        self.L = np.zeros(N)
        self.scalar = np.zeros(N) # Scalar, gotta rename it
        self.Sm = np.zeros(N)
        self.Sh = np.zeros(N)
        self.nu_t = np.zeros(N) # Turbulent viscosity
        self.Kq = np.zeros(N) # Turbulent diffusivity for q2
        self.Kz = np.zeros(N) # Turbulent scalar diffusivity

        # self.Cveg = (np.linspace(0.0004, 0.0006, N)) #0.0005*np.ones(N) # Variable drag coefficient
        self.Cveg = np.zeros(N)
        # Model Parameters
        self.H = H # Column depth (m)
        self.Length = Len # Plane Length (m)
        self.N = N  # No. of grid points
        
        self.dz = H/N
        self.dx = self.Length/N

        self.M = 1000  # No. of time steps
        
        # Dynamically adjusting dt for stability
        if (self.dz)**2/1e-6 <= 60:
            self.dt = (self.dz)**2/1e-6
        else:
            self.dt = 60  # Size of time step (s)

        self.beta = self.dt/(self.dz**2)

    def import_veg(self, alpha, density, height):
        # compute relationship between LAI/density readings
        # and Cveg here then send to column
        iid = int(height/self.dz)
        self.Cveg[:iid] = alpha*0.0001*density # Please have alpha between 0 and 1
        return self
        
    def setup(self, A, B, C, Sq, kappa, SMALL, nu, g, rho0, alpha):
        A1 = A[0]
        A2 = A[1]
        B1 = B[0]
        B2 = B[1]
        C1 = C[0]

        # Initializing Grid
        for i in range(self.N): ##### In the future separate N in Nx and Nz
            self.z[i] = -self.H + self.dz*((i+1) - 0.5)
        for i in range(self.N): ##### In the future separate N in Nx and Nz
            self.x[i] = self.Length - self.dx*((i+1) - 0.5)
        # Temperature Distribution
        de1C = 5 # Change in temperature at initial thermocline
        zde1C = -5 # Position of initial thermocline
        dzde1C = -2 # Width of initial thermocline 
        T = 15
        for i in range(self.N):
            self.scalar[i] = T
            if self.z[i] <= zde1C - 0.5*dzde1C:
                self.scalar[i] = T
            elif self.z[i] >= zde1C + 0.5*dzde1C:
                self.scalar[i] = T + de1C
            else:
                self.scalar[i] = T + de1C*(self.z[i] - zde1C + 0.5*dzde1C) / dzde1C
            self.rho[i] = rho0*(1-alpha*(self.scalar[i] - T))
        
        # Density distribution from Temperature
        self.rho = rho0*(1-alpha*(self.scalar - T))

        # Brunt-Vaisala Frequency from density profile

                # Here in the MATLAB code, we would sometimes see N2 being negative, leading to 
        # a complex value for N_BV. MATLAB is well accustomed to complex values and pursues calculations 
        # anyway. However this raises a math error in Python as taking the sqrt of a negative number in
        # Python is strictly prohibited. We set N_BV to 0 in those cases to not "lose" mixing in those cases,
        # and to, well, keep the code running. We use try/catch here to switch to zero in case the error is raised.
        try:
            self.N_BV[0] = math.sqrt(((-g/rho0)*(self.rho[1]-self.rho[0])/(self.dz)))
            self.N_BVsq[0] = ((-g/rho0)*(self.rho[1]-self.rho[0])/(self.dz))
        except ValueError:
            self.N_BV[0] = math.sqrt(((-g/rho0)*(self.rho[0]-self.rho[1])/(self.dz)))
            self.N_BVsq[0] = ((-g/rho0)*(self.rho[1]-self.rho[0])/(self.dz))
        for i in range(1,self.N-1):
            try: 
                self.N_BV[i] = math.sqrt(((-g/rho0)*(self.rho[i+1] - self.rho[i])/(self.dz)))
                self.N_BVsq[i] = ((-g/rho0)*(self.rho[i+1] - self.rho[i])/(self.dz))
            except ValueError:
                self.N_BV[i] = math.sqrt(((-g/rho0)*(self.rho[i] - self.rho[i+1])/(self.dz)))
                self.N_BVsq[i] = ((-g/rho0)*(self.rho[i+1] - self.rho[i])/(self.dz))
        try: 
            self.N_BV[-1] = math.sqrt(((-g/rho0)*(self.rho[self.N-1] - self.rho[self.N-2])/(self.dz)))
            self.N_BVsq[-1] = ((-g/rho0)*(self.rho[self.N-1] - self.rho[self.N-2])/(self.dz))
        except ValueError:
            self.N_BV[-1] = math.sqrt(((-g/rho0)*(self.rho[self.N-2] - self.rho[self.N-1])/(self.dz)))
            self.N_BVsq[-1] = ((-g/rho0)*(self.rho[self.N-1] - self.rho[self.N-2])/(self.dz))


        # Initial Conditions
        for i in range(self.N):
            self.Q[i] = math.sqrt(self.Q2[i])
            self.L[i] = -kappa*self.H*(self.z[i]/self.H)*(1-(self.z[i]/self.H))
            Gh = -self.N_BVsq[i]*((self.L[i])/(self.Q[i] + SMALL))**2
            Gh = min(Gh, 0.0233)
            Gh = max(Gh, -0.28)        
            num= B1**(-1/3) - A1*A2*Gh*((B2-3*A2)*(1-6*A1/B1)-3*C1*(B2+6*A1))
            dem= (1-3*A2*Gh*(B2+6*A1))*(1-9*A1*A2*Gh)
            self.Sm[i] = num/dem
            self.Sh[i] = A2*(1-6*A1/B1)/(1-3*A2*Gh*(B2+6*A1))
            self.nu_t[i] = self.Sm[i] * self.Q[i] * self.L[i] + nu # Turbulent diffusivity for Q2
            self.Kq[i] = Sq*self.Q[i]*self.L[i] + nu # Turbulent viscosity
            self.Kz[i] = self.Sh[i] * self.Q[i] * self.L[i] + nu # Turbulent scalar diffusivity

        return self