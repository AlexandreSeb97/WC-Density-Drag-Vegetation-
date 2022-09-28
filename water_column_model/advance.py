# Time advancement functions
# Each function performs a single time-step for a corresponding component
# adv function calls all function to perform a time-step for all components
# Components are:
# u, v, temp, rho, q2, q21, l, kz, nu_t, kq

import math
import numpy as np
from .params import A, B, C, E, SMALL, C_D, kappa, rho0, g, alpha, zb, u_crit, nu, Sq

def TDMA(aX, bX, cX, dX, N):
    x = np.zeros(N)
    for i in range(1, N):
        bX[i] = bX[i] - aX[i]/bX[i-1]*cX[i-1]
        dX[i] = dX[i] - aX[i]/bX[i-1]*dX[i-1]
    x[-1] = dX[-1]/bX[-1]
    for i in range(N-2, -1, -1):
        x[i] = (1/bX[i])*(dX[i] - cX[i]*x[i+1])
    return x

def wc_advance(c, T_Px, Px0, t):
    #  Place previous variable f into fp (i.e. q2 into q2p, etc)
    N = c.N
    dz = c.dz
    H = c.H
    beta = c.beta
    dt = c.dt

    [Up, Vp, nu_tp, Q2p, Q2Lp, Lp, Kqp, Kzp, N_BVp, N_BVsqp, Cp] = previous(c)
    ustar = abs(Up[0]*math.sqrt(C_D))
    Px = pressure(N, T_Px, Px0, t)
    [Smnew, Shnew, Ghnew] = update_params(c, SMALL, A, B, C)
    Unew = velocity(c, N, kappa, beta, dt, Px, Up, nu_tp)
    [Cnew, N_BVnew, N_BVsqnew, rhonew] = scalar_trans(c, beta, Kzp, Cp, dz, g, rho0, alpha)
    Q2new = turb_q2(c, N, beta, dt, B, ustar, Up, nu_tp, Kqp, Kzp, Lp, N_BVsqp, Q2p, SMALL)
    Q2Lnew = turb_q2l(c, N, beta, dt, B, E, ustar, kappa, H, Up, nu_tp, zb, Kqp, Kzp, Q2Lp, Lp, N_BVsqp, Q2p, SMALL)
    [Kqnew, Kznew, nu_tnew, Qnew, Q2Lnew, Lnew] = turbmix(c, SMALL, zb, Sq, Q2new, Q2Lnew, Smnew, Shnew, nu, N_BVsqp)
    
    return Unew, Cnew, Qnew, Q2new, Q2Lnew, rhonew, Lnew, nu_tnew, Kznew, Kqnew, N_BVnew, N_BVsqnew

def previous(c):
    Cp = c.scalar
    Q2p = c.Q2
    Q2Lp = c.Q2L
    Lp = c.L
    Kzp = c.Kz
    nu_tp = c.nu_t
    Kqp = c.Kq
    N_BVp = c.N_BV
    N_BVsqp = c.N_BVsq
    Up = c.U
    Vp = c.V
    return Up, Vp, nu_tp, Q2p, Q2Lp, Lp, Kqp, Kzp, N_BVp, N_BVsqp, Cp
   
def pressure(N, T_Px, Px0, t):
    Px = np.ones(N)
    #  Update pressure forcing term for the current timestep
    if T_Px == 0.0:
        Px = Px*Px0  # Steady and constant forcing for now
    else:
        Px = Px*Px0*math.cos(2*math.pi*t/(3600*T_Px))
    return Px

def update_params(c, SMALL, A, B, C):
    A1 = A[0]
    A2 = A[1]
    B1 = B[0]
    B2 = B[1]
    C1 = C[0]

    # Update parameters for the model, Sm and Sh
    Smnew = np.zeros(c.N)
    Shnew = np.zeros(c.N)
    for i in range(c.N):		
        Gh=-c.N_BVsq[i]*(c.L[i]/(c.Q[i]+SMALL))**2 
        # set LIMITER for Gh 
        Gh=min(Gh, 0.0233)
        Gh=max(Gh, -0.28)
        num=B1**(-1/3)-A1*A2*Gh*((B2-3*A2)*(1-6*A1/B1)-3*C1*(B2+6*A1))
        dem=(1-3*A2*Gh*(B2+6*A1))*(1-9*A1*A2*Gh)
        #L = kappa*(z+ H)*(-z/H)**0.5
        Smnew[i]=num/dem
        Shnew[i]=A2*(1-6*A1/B1)/(1-3*A2*Gh*(B2+6*A1)) 

    return Smnew, Shnew, Gh

def velocity(c, N, kappa, beta, dt, Px, Up, nu_tp):
    aU = np.zeros(N)
    bU = np.zeros(N)
    cU = np.zeros(N)
    dU = np.zeros(N)
    #### NEW ####
    # Advance velocity (U, could also implement the same code for V)
    for i in range(1, N-1):
        aU[i] = -0.5*beta*(nu_tp[i] + nu_tp[i-1])
        bU[i] = 1+0.5*beta*(nu_tp[i+1] + 2*nu_tp[i] + nu_tp[i-1])-(dt*c.Cveg[i]*Up[i])
        cU[i] = -0.5*beta*(nu_tp[i] + nu_tp[i+1])
        dU[i] = Up[i] - dt*Px[i]
    # Bottom-Boundary: log-law
    bU[0] = 1+0.5*beta*(nu_tp[1] + nu_tp[0] + 2*(math.sqrt(C_D)/kappa)*nu_tp[0])-(dt*c.Cveg[0]*Up[0])
    cU[0] = -0.5*beta*(nu_tp[1] + nu_tp[0])
    dU[0] =  Up[0] - dt*Px[0]
    # Top-Boundary: no stress
    aU[-1] = -0.5*beta*(nu_tp[-1] + nu_tp[N-2])
    bU[-1] = 1+0.5*beta*(nu_tp[-1] + nu_tp[N-2])-(dt*c.Cveg[-1]*Up[-1])
    dU[-1] = Up[-1] - dt*Px[-1]
    # Thomas algorithm to solve for U
    Unew = TDMA(aU, bU, cU, dU, N)

    return Unew

def scalar_trans(c, beta, Kzp, Cp, dz, g, rho0, alpha):
    aC = np.zeros(c.N)
    bC = np.zeros(c.N)
    cC = np.zeros(c.N)
    dC = np.zeros(c.N)
    N_BVnew = np.zeros(c.N)
    N_BVsq = np.zeros(c.N)
    rhonew = np.zeros(c.N)

    # Advance scalars/density (C, rho) 
    for i in range(1, c.N-1):
        aC[i] = -0.5*beta*(Kzp[i] + Kzp[i-1])
        bC[i] = 1+0.5*beta*(Kzp[i+1] + 2*Kzp[i] + Kzp[i-1])
        cC[i] = -0.5*beta*(Kzp[i] + Kzp[i+1])
        dC[i] = Cp[i]

    
    # Bottom-Boundary: no flux for scalars
    bC[0] = 1+0.5*beta*(Kzp[1] + Kzp[0])
    cC[0] = -0.5*beta*(Kzp[1] + Kzp[0])
    dC[0] =  Cp[0]
    # Top-Boundary: no flux for scalars
    aC[-1] = -0.5*beta*(Kzp[-1] + Kzp[c.N-2])
    bC[-1] = 1+0.5*beta*(Kzp[-1] + Kzp[c.N-2])
    dC[-1] = Cp[-1]
    # Thomas algorithm to solve for C
    Cnew = TDMA(aC, bC, cC, dC, c.N)
    #update density and Brunt-Vaisala frequency
    for i in range(c.N):
        rhonew[i] = rho0*(1-alpha*(Cnew[i] - 15)) # Change 15 to T
    try: 
        N_BVnew[0] = math.sqrt((((-g/rho0)*(rhonew[1]-rhonew[0])/dz)))
        N_BVsq[0] = (((-g/rho0)*(rhonew[1]-rhonew[0])/dz))
    except ValueError:
        N_BVnew[0] = math.sqrt((((-g/rho0)*(rhonew[0]-rhonew[1])/dz)))
        N_BVsq[0] = (((-g/rho0)*(rhonew[1]-rhonew[0])/dz))
    for i in range(1, c.N-1):
        try: 
            N_BVnew[i] = math.sqrt(((-g/rho0)*(rhonew[i+1] - rhonew[i])/(dz)))
            N_BVsq[i] = ((-g/rho0)*(rhonew[i+1] - rhonew[i])/(dz))
        except ValueError:
            N_BVnew[i] = math.sqrt(((-g/rho0)*(rhonew[i] - rhonew[i+1])/(dz)))
            N_BVsq[i] = ((-g/rho0)*(rhonew[i+1] - rhonew[i])/(dz))
    try: 
        N_BVnew[-1] = math.sqrt(((-g/rho0)*(rhonew[-1] - rhonew[c.N-2])/dz))
        N_BVsq[-1] = ((-g/rho0)*(rhonew[-1] - rhonew[c.N-2])/dz)
    except ValueError:
        N_BVnew[-1] = math.sqrt(((-g/rho0)*(rhonew[c.N-2] - rhonew[-1])/dz))
        N_BVsq[-1] = ((-g/rho0)*(rhonew[-1] - rhonew[c.N-2])/dz)

    return Cnew, N_BVnew, N_BVsq, rhonew
    

def turb_q2(c, N, beta, dt, B, ustar, Up, nu_tp, Kqp, Kzp, Lp, N_BVsqp, Q2p, SMALL):
    B1 = B[0]
    aQ2 = np.zeros(N)
    bQ2 = np.zeros(N)
    cQ2 = np.zeros(N)
    dQ2 = np.zeros(N)
    #  Advance turbulence parameters (q2, q2l - q2 first, then q2l)
    for i in range(1, N-1):
            diss = 2 * dt *np.sqrt(Q2p[i])/(B1*Lp[i]) # Coefficient for linearized term
            aQ2[i] = -0.5*beta*(Kqp[i] + Kqp[i-1])
            bQ2[i] = 1+0.5*beta*(Kqp[i+1] + 2*Kqp[i] + Kqp[i-1]) + diss
            cQ2[i] = -0.5*beta*(Kqp[i] + Kqp[i+1])
            dQ2[i] = Q2p[i] + 0.25*beta*nu_tp[i]*(Up[i+1]-Up[i-1])**2 -dt*Kzp[i]*(N_BVsqp[i])
    # Bottom-Boundary Condition 
    Q2bot = B1**(2/3) * ustar**2
    bdryterm = 0.5*beta*Kqp[0]*Q2bot
    diss =  2 * dt *(np.sqrt(Q2p[0])/(B1*Lp[0]))
    bQ2[0] = 1+0.5*beta*(Kqp[1] + Kqp[0]) + diss
    cQ2[0] = -0.5*beta*(Kqp[1] + Kqp[0])
    dQ2[0] = Q2p[0] + dt*((ustar**4)/nu_tp[0]) - dt*Kzp[0]*(N_BVsqp[0]) + bdryterm
    # Top-Boundary Condition
    diss =  2 * dt *(np.sqrt(Q2p[-1])/(B1*Lp[-1]))
    aQ2[-1] = -0.5*beta*(Kqp[-1] + Kqp[N-2])
    bQ2[-1] = 1+0.5*beta*(Kqp[-1] + 2*Kqp[-1] + Kqp[N-2]) + diss
    dQ2[-1] = Q2p[-1] + 0.25*beta*nu_tp[-1]*((Up[-1] - Up[N-2])**2) -4*dt*Kzp[-1]*(N_BVsqp[-1])
    # TDMA to solve for q2
    Q2new = TDMA(aQ2, bQ2, cQ2, dQ2, N)
    # Kluge to prevent negative values from causing instabilities
    for i in range(N):
        if Q2new[i] < SMALL:
            Q2new[i] = SMALL
    return Q2new
    
def turb_q2l(c, N, beta, dt, B, E, ustar, kappa, H, Up, nu_tp, zb, Kqp, Kzp, Q2Lp, Lp, N_BVsqp, Q2p, SMALL):
    B1 = B[0]
    E1 = E[0]
    E2 = E[1]
    E3 = E[2]

    aQ2L = np.zeros(N)
    bQ2L = np.zeros(N)
    cQ2L = np.zeros(N)
    dQ2L = np.zeros(N)
    for i in range(1, N-1):
            diss = 2*dt*((Q2p[i]**0.5) / (B1*Lp[i]))*(1+E2*(Lp[i]/(kappa*abs(-H-c.z[i])))**2 + E3*(Lp[i]/(kappa*abs(c.z[i])))**2)
            aQ2L[i] = -0.5*beta*(Kqp[i] + Kqp[i-1])
            bQ2L[i] = 1+0.5*beta*(Kqp[i+1] + 2*Kqp[i] + Kqp[i-1]) + diss
            cQ2L[i] = -0.5*beta*(Kqp[i] + Kqp[i+1])
            dQ2L[i] = Q2Lp[i] + 0.25*beta*nu_tp[i]*E1*Lp[i]*(Up[i+1]-Up[i-1])**2 - 2*dt*Lp[i]*E1*Kzp[i]*(N_BVsqp[i])
    # Bottom-Boundary Condition
    q2lbot = B1**(2/3) * (ustar**2) * kappa * zb
    bdryterm = 0.5*beta*Kqp[0]*q2lbot
    diss =  2 * dt *(Q2p[0]**0.5)/(B1*Lp[0])*(1+E2*(Lp[0]/(kappa*abs(-H-c.z[0])))**2 + E3*(Lp[0]/(kappa*abs(c.z[0])))**2)
    bQ2L[0] = 1+0.5*beta*(Kqp[1] + Kqp[0]) + diss
    cQ2L[0] = -0.5*beta*(Kqp[1] + Kqp[0])
    dQ2L[0] = Q2Lp[0] + dt*((ustar**4)/nu_tp[0])*E1*Lp[0] - dt*Lp[0]*E1*Kzp[0]*(N_BVsqp[0]) + bdryterm
    # Top-Boundary Condition
    diss =  2 * dt *(Q2p[-1]**0.5)/(B1*Lp[-1])*(1+E2*(Lp[-1]/(kappa*abs(-H-c.z[-1])))**2 + E3*(Lp[-1]/(kappa*abs(c.z[-1])))**2)
    aQ2L[-1] = -0.5*beta*(Kqp[-1] + Kqp[N-2])
    bQ2L[-1] = 1+0.5*beta*(Kqp[-1] + 2*Kqp[-1] + Kqp[N-2]) + diss # Are we using kq or kqp here?
    dQ2L[-1] = Q2Lp[-1] + 0.25*beta*nu_tp[-1]*E1*Lp[-1]*(Up[-1]-Up[N-2])**2 - 2*dt*Lp[-1]*E1*Kzp[-1]*(N_BVsqp[-1])
    # TDMA to solve for q2
    Q2Lnew = TDMA(aQ2L, bQ2L, cQ2L, dQ2L, N)
    # Making sure to prevent negative values
    for i in range(N):
        if Q2Lnew[i] < SMALL:
            Q2Lnew[i] = SMALL
    return Q2Lnew

def turbmix(c, SMALL, zb, Sq, Q2new, Q2Lnew, Sm, Sh, nu, N_BVsq):
    #  Calculate turbulent lengthscale (l) and mixing coefficients (kz, nu_t, kq)
    #     Works will all updated values 
    Lnew = np.zeros(c.N)
    Kznew = np.zeros(c.N)
    Kqnew = np.zeros(c.N)
    nu_tnew = np.zeros(c.N)
    
    Qnew = np.zeros(c.N)#np.sqrt(Q2new)
    for i in range(c.N):
        Qnew[i] = math.sqrt(Q2new[i])
        Lnew[i] = Q2Lnew[i]/(Q2new[i] + SMALL)
        
        # Limit due to stable stratification
        if (Lnew[i]**2)*(N_BVsq[i]) > 0.281*Q2new[i]:
            # Adjust Q2L as well as L
            Q2Lnew[i] = Q2new[i]*math.sqrt(0.281*Q2new[i]/(N_BVsq[i] + SMALL))
            Lnew[i] = Q2Lnew[i] / Q2new[i]  
        # Keep L from becoming zero -- zb=bottom roughness parameter
            
        if abs(Lnew[i]) <= zb:
            Lnew[i] = zb
        # Update diffusivities 
        Kqnew[i] = Sq*Qnew[i]*Lnew[i] + nu
        nu_tnew[i] = Sm[i]*Qnew[i]*Lnew[i] + nu
        Kznew[i] = Sh[i]*Qnew[i]*Lnew[i] + nu  
   
    return Kqnew, Kznew, nu_tnew, Qnew, Q2Lnew, Lnew    