import os
import sys
sys.path.append('../')
import numpy as np
from pdb import set_trace
from matplotlib import rc as matplotlibrc
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from itertools import product, combinations
from scipy import signal
import pickle
from scipy.signal import argrelextrema
import parameters as para
from matplotlib import cm
figheight      = 8*0.6
figwidth       = 6*0.6
lineWidth      = 1.5
textFontSize   = 18
gcafontSize    = 15*1.5
legendFontSize = 20
headwidth	   = 0.03

N 			= para.N
rpkl_name	= './../data/data_one.pkl'
data 		= pickle.load( open( rpkl_name, "rb" ) )#,encoding="latin1"
tlist    	= data['tlist']		
ylist    	= data['ylist']			
fftlist 	= data['fftlist']	



Nf          = len(tlist)
Np 			= ylist.shape[1]

x 			= np.linspace(0,1,Np)*np.pi*2


tt,xx=np.meshgrid(x,tlist)

fig = plt.figure(0, figsize=(figwidth*1.,figheight*1.0))
ax=fig.add_subplot(1,1,1)

plt.contourf(tt[:,:],xx[:,:],ylist[:],300,cmap=cm.Blues)
plt.tight_layout()

fname = 'brain.png'
plt.savefig('./../figures/'+fname)
print(fname,' saved!')
plt.show()
plt.close()	