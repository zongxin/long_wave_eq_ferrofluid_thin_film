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
# import functions as func
from scipy.signal import hilbert
import parameters as para
# matplotlibrc('text.latex', preamble=r'\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

figheight      = 4
figwidth       = 5
lineWidth      = 1.5
textFontSize   = 18
gcafontSize    = 15
legendFontSize = 20


figname 	='vf.png'

N 			= para.N
rpkl_name	= './../data/data_one.pkl'
data 		= pickle.load( open( rpkl_name, "rb" ) )#,encoding="latin1"

tlist    	= data['tlist']		
ylist    	= data['ylist']			
fftlist 	= data['fftlist']	
# set_trace()
Nf 			= len(tlist)
Nm 			= 4
yhat 		= np.zeros((Nf,Nm ))
m 			= np.array(range(Nm ))
kk=4

phase=np.zeros((Nf,Nm ))
w_list=np.zeros((Nf,Nm ))
vf_list=np.zeros((Nf,Nm ))
Sk=np.zeros(Nf)
time=np.zeros(Nf)


for i in range (0,Nf):
	phase[i,m]=np.angle(fftlist[i,(m+1)*kk])



for im in range(Nm ):
	phase1=phase[0:-1,im]
	phase2=phase[1:,im]
	a=np.where(np.abs(phase1-phase2)>1.)[0]
	# set_trace()
	for ia,aa in enumerate(a):
		phase[aa+1:,im]=phase[aa+1:,im]-2*np.pi
	w_list[:,im]=-np.gradient(phase[:,im],tlist)
	vf_list[:,im]=w_list[:,im]/((im+1)*kk) 

# set_trace()
fig = plt.figure(0,figsize=(figwidth*2.0,figheight*0.8))

ax  = fig.add_subplot(121,alpha=1)#
for im in range(Nm ):
	# plt.plot(phase[:,im],'.')
	plt.plot(tlist,w_list[:,im])	
	# plt.scatter(a,phase[a,im])

# # ax.set_ylim([0,5000])
# ax.set_xlabel(r'$t$',fontsize=1.2*gcafontSize)
# ax.set_ylabel(r'$d\omega/dt$',fontsize=1.2*gcafontSize)
plt.setp(ax.get_xticklabels(),fontsize=1.2*gcafontSize)
plt.setp(ax.get_yticklabels(), fontsize=1.2*gcafontSize)
# ax.set_xticks([0,0.01,0.02,0.03])

ax  = fig.add_subplot(122,alpha=1)#
for im in range(Nm ):
	plt.plot(tlist,vf_list[:,im])



time_string=r'$v_f=$'+str(round(np.mean(vf_list[-3,:]),5))
# print ('v_f=',np.mean(vf_list[:,-3]))
ax.text(0.03,220,time_string)
ax.set_ylim([100,300])
ax.set_xlabel(r'$t$',fontsize=1.2*gcafontSize)
ax.set_ylabel(r'$v_f$',fontsize=1.2*gcafontSize)
plt.setp(ax.get_xticklabels(),fontsize=1.2*gcafontSize)
plt.setp(ax.get_yticklabels(), fontsize=1.2*gcafontSize)
# ax.set_xticks([0,0.01,0.02,0.03])

plt.tight_layout()	
folder='./../figures/'	
plt.savefig(folder+figname)
plt.show()
set_trace()
plt.close()
print (figname," saved!")




# plt.plot(time,As)
# plt.plot(time,Sk)

# plt.show()

# pkl_name='./post_proc/H_NB_1_37.pkl'
# data={}

# data['time']	= time
# data['As']		= As		
# data['Sk']		= Sk			
# output = open(pkl_name, 'wb')
# pickle.dump(data, output)
# output.close()
# print pkl_name," saved!"

