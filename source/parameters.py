import numpy as np
import sys
from pdb import set_trace

# physical parameters
rho  	= 80
q   	= 0.01
alpha	= -2*rho*(2-3*q)
gamma 	= rho*(1-2*q)
beta 	= 2*rho*(q*(1-q))**0.5
delta 	= 0.1

# simulation parameters
N       = 512
Ttot    = 0.06
dt      = 5e-8
Data_output=2000
Plot_output=20000

# domain size
Lx      = 2*np.pi
x       = np.linspace(0, Lx, N, endpoint=False)

# Initialization
wavek   = np.fft.fftfreq(N)*N*2*np.pi/Lx
wavek[int(N / 2)] = 0

u0      = 0.01*np.sin(4*x)

#	Purturbation
Perturbation = 0
Basename 	 = "base.pkl"
up      	 = 0.1*0.2*np.sin(4.*x)
 

#Restart
Restart 	  = 0
Res_i		  = 5
Resname       = "./data/data_"+str(Res_i)+".pkl"
Extratime 	  = 0.03



