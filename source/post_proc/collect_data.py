import os
import sys
sys.path.append('../')
import numpy as np
from pdb import set_trace
from matplotlib import rc as matplotlibrc
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from scipy import signal
import pickle
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft
import parameters as para
# matplotlibrc('text.latex', preamble=r'\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

figheight      = 4
figwidth       = 4
lineWidth      = 1.5
textFontSize   = 18
gcafontSize    = 15
legendFontSize = 20

alpha   = para.alpha
gamma   = para.gamma
beta    = para.beta
delta   = para.delta

N       = para.N
x       = para.x
dx      = x[2]-x[1]
wavek   = para.wavek


Nf  = 600
Nf  = Nf+1

ylist   = np.zeros((Nf,N))
fftlist = np.zeros((Nf,N),dtype=complex)
tlist   = np.zeros((Nf))

for i in range(Nf):
    dataname='./../data/data_'+str(i)+'.pkl'
    data             = pickle.load( open( dataname, "rb" ) )
    # set_trace()
    tlist[i]         = data['time']#+dt          
    ylist[i,:]       = data['y'] 
    fftlist[i,:]     = data['fft'] 



pkl_name='./../data/data_one.pkl'
data={}
data['alpha']       = alpha
data['beta']        = beta
data['gamma']       = gamma
data['delta']       = delta
data['tlist']       = tlist  
data['ylist']       = ylist 
data['fftlist']     = fftlist 
output = open(pkl_name, 'wb')
pickle.dump(data, output)
output.close()
print (pkl_name," saved!")

