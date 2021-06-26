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
# from matplotlib import rcParams, cycler

# matplotlibrc('text.latex', preamble=r'\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

figheight      = 8
figwidth       = 8
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




fig = plt.figure(0, figsize=(figwidth*1.,figheight*1.0))
ax=fig.add_subplot(1,1,1)

for i in range (0,Nf):

	
	ylist[i,:]=ylist[i,:]*5
	
	if (i%5==0):

		plt.plot(x,ylist[i,:]+np.int(i/5)*1.,color='k',lw=lineWidth*0.8)



	
ax.set_xlim([-0.,np.pi*2])
ax.set_xticks([0,np.pi,np.pi*2])
ax.set_xticklabels(['0', r'$\pi$', r'2$\pi$'],fontsize=1.2*gcafontSize)
# ax.set_ylim([-1.,70*1.1])	
# ax.set_yticks([0.,20.*1.1,40*1.1,60*1.1])	
# ax.set_yticklabels([r'$0$',r'$0.02$',r'$0.04$',r'$0.06$'],fontsize=1.2*gcafontSize)
# # ax.set_yticks([])


ax.set_xlabel(r'$\theta$',fontsize=1.2*gcafontSize)
ax.set_ylabel(r'$t$',fontsize=1.2*gcafontSize)
# ax.tick_params()
# ax.spines['left'].set_visible(False)	
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['left'].set_axisline_style("-|>")
# ax.axis('off')				
figname = "wave.png"
plt.tight_layout()
plt.savefig('./../figures/'+figname)
plt.show()
plt.close()	



