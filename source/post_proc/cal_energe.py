import os
import sys
sys.path.append('../')
import numpy as np
from pdb import set_trace
from matplotlib import rc as matplotlibrc
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from itertools import product, combinations
from scipy import signal
import pickle
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import parameters as para
from scipy.fftpack import fft, ifft
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')


figheight      = 4
figwidth       = 4
lineWidth      = 1.5
textFontSize   = 18
gcafontSize    = 15
legendFontSize = 20


figname 	='energy_dt.png'

N 			= para.N
rpkl_name	= './../data/data_one.pkl'
data 		= pickle.load( open( rpkl_name, "rb" ) )#,encoding="latin1"

tlist    	= data['tlist']	
ylist    	= data['ylist']			
fftlist 	= data['fftlist']	

Nf 			= len(tlist)
Nm 			= 5
yhat 		= np.zeros((Nf,Nm ))
m 			= np.array(range(Nm ))
kk=4

for i in range (0,Nf):
	yhat[i,m]=np.abs(fftlist[i,(m+1)*kk])/N
	

fig = plt.figure(0,figsize=(figwidth*1.4,figheight))
ax  = fig.add_subplot(111,alpha=1)#

for im in range (Nm ):
	plt.plot(tlist,yhat[:,im],label=str((im+1)*kk))
ax.grid('on')	

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels,fontsize=gcafontSize,mode='expanded',ncol=3,loc='best',numpoints=1)	
plt.xlabel(r"$t$",fontsize=1.0*gcafontSize)
plt.ylabel(r"$\sqrt{\xi_k\xi_{-k}}$",fontsize=1.0*gcafontSize)

plt.setp(ax.get_xticklabels(),fontsize=1*gcafontSize)
plt.setp(ax.get_yticklabels(), fontsize=1*gcafontSize)

plt.tight_layout()	
folder='./../figures/'	
plt.savefig(folder+figname)
plt.show()
plt.close()
print (figname," saved!")






