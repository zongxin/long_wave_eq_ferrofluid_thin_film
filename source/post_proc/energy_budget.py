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



def cal_der(y):
    y_hat = fft(y)
    Yx      =  1j * wavek    * y_hat
    Yxx     = -     wavek**2 * y_hat
    Yxxx    = -1j * wavek**3 * y_hat
    Yxxxx   =       wavek**4 * y_hat            
    yx      = np.real(ifft(Yx))
    yxx     = np.real(ifft(Yxx))
    yxxx    = np.real(ifft(Yxxx))
    yxxxx   = np.real(ifft(Yxxxx))        
    return yx,yxx,yxxx,yxxxx 

def cal_energy(y,yx,yxx,yxxx,yxxxx,dx):
	e1 = np.sum(-delta*alpha*yx**2)*dx
	# set_trace()
	e2 = np.sum(-yxx**2)*dx
	e3 = np.sum(-delta*alpha*yx**2*y)*dx	
	e4 = np.sum(0.5*beta*yx**3)*dx
	e5 = np.sum(-yxx**2*y)*dx		
	return np.array([e1,e2,e3,e4,e5])



dataname='./../data/data_one.pkl'
data        = pickle.load( open( dataname, "rb" ) )
tlist       = data['tlist']#+dt          
ylist       = data['ylist'] 
Nf          = len(tlist)
elist       = np.zeros((Nf,5))
for i in range(Nf):
    y                   = ylist[i,:]
    yx,yxx,yxxx,yxxxx   = cal_der(y)
    elist[i,:]          = cal_energy(y,yx,yxx,yxxx,yxxxx,dx)

    print('--------------')
    print('time:',tlist[i])    
    print('Etot:',np.sum(elist[i,:]))






fig = plt.figure(0, figsize=(figwidth*1.4,figheight))
ax  = fig.add_subplot(111,alpha=1)#

plt.plot(tlist,elist[:,0],c='tab:red',label=r'$\alpha$') # alpha

plt.plot(tlist,elist[:,1],c='tab:green',label='surf') # surf
plt.plot(tlist,elist[:,2],c='tab:red',alpha=0.7) # alpha-nl
plt.plot(tlist,elist[:,3],c='tab:blue',label=r'$\beta$-nl',alpha=0.7) # beta-nl 
plt.plot(tlist,elist[:,4],c='tab:green',alpha=0.7) # surf-nl
         
plt.plot(tlist,np.sum(elist,axis=1),c='k')      
plt.axhline(0,c='grey',ls='--')
etot=np.sum(elist,axis=1)[-1]
string='etot='+"{:e}".format(etot)
ax.text(0.004,150,string)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels,fontsize=gcafontSize,loc='best',numpoints=1) 

plt.xlabel(r"$t$",fontsize=1.0*gcafontSize)
plt.ylabel(r"$<\dot{\eta}^3>$",fontsize=1.0*gcafontSize)
plt.tight_layout() 
fname = './../figures/energy_budget.png'
plt.savefig(fname)
print(fname,' saved!')
plt.show()
plt.close()
    # plt.show()


#=====================================================
# pkl_name='shooting_fa9.pkl'
# data={}
# data['f']     = y_plot
                
# output = open(pkl_name, 'wb')
# pickle.dump(data, output)
# output.close()

