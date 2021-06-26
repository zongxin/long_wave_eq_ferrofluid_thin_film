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
# from scipy.integrate import solve_bvp
# from scipy.integrate import solve_ivp
import pickle
from scipy.interpolate import interp1d

from scipy.signal import hilbert

# matplotlibrc('text.latex', preamble=r'\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

figheight      = 4
figwidth      = 4
lineWidth      = 1.5
textFontSize   = 18
gcafontSize    = 15
legendFontSize = 20


def cal_mean(f,da):
	return np.sum(f)*da/(2*np.pi)



figname='skew_assym.png'


rpkl_name	= './../data/data_one.pkl'
data 		= pickle.load( open( rpkl_name, "rb" ) )#,encoding="latin1"

tlist 	= data['tlist']				
ylist 	= data['ylist']	

Nf= len(tlist)
As=np.zeros(Nf)
Sk=np.zeros(Nf)

for i in range (Nf):
	y  	= ylist[i,:]
	fss = y
	fm 	= np.mean(y)
	f_p	= fss-fm

	Sk[i]=np.mean(f_p**3)/np.mean(f_p**2)**1.5
	hilb=np.imag(hilbert(f_p))
	As[i]=np.mean(hilb**3)/np.mean(f_p**2)**1.5

	# set_trace()
fig = plt.figure(0,figsize=(figwidth*1.4,figheight))	
ax  = fig.add_subplot(111,alpha=1)#
plt.plot(tlist,As)
plt.plot(tlist,Sk)

time_string=r'$A_s=$'+str(As[-2])
ax.text(0.004,0.1,time_string)
time_string=r'$S_k=$'+str(Sk[-2])
ax.text(0.004,0.,time_string)
ax.set_xlabel(r'$t$',fontsize=1.*gcafontSize)
ax.set_ylabel(r'$A_s,S_k$',fontsize=1.*gcafontSize)

folder='./../figures/'
plt.tight_layout()		
plt.savefig(folder+figname)
plt.show()
plt.close()
print (figname," saved!")


# pkl_name=nname+'.pkl'
# data={}

# data['time']	= time
# data['As']		= As		
# data['Sk']		= Sk			
# output = open(pkl_name, 'wb')
# pickle.dump(data, output)
# output.close()
# print (pkl_name," saved!")

