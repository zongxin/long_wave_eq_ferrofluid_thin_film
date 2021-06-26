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
# import scipy.sparse as spar
# matplotlibrc('text.latex', preamble=r'\usepackage{color}')
matplotlibrc('text',usetex=True)
matplotlibrc('font', family='serif')

figheight      = 4
figwidth       = 4
lineWidth      = 1.5
textFontSize   = 18
gcafontSize    = 15
legendFontSize = 20

alpha   = -para.alpha
gamma   = para.gamma
beta    = para.beta
delta   = para.delta

N       = para.N
x       = para.x
dx      = x[2]-x[1]
wavek   = para.wavek
vff=218.0287


# vff=20
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


jlist=np.linspace(1.,N-1.,N-1)

D1=np.zeros((N,N))
col=np.zeros(N)
col[0]=0.
col[1:]= 0.5*(-1)**jlist/np.tan(jlist*dx/2)
for i in range(N):
  D1[:,i]=np.roll(col, i)


D2=np.zeros((N,N))
col=np.zeros(N)
col[0]= 1./12*(-N*N-2)
col[1:]= 0.5*(-1)**(jlist+1)/np.sin(jlist*dx/2)**2
for i in range(N):
  D2[:,i]=np.roll(col, i)

D3=np.zeros((N,N))
col=np.zeros(N)
col[0]= 0.
col[1:]= 1./8*(-1)**(jlist+1)/np.tan(jlist*dx/2)*(N**2-6./np.sin(jlist*dx/2)**2)
for i in range(N):
  D3[:,i]=np.roll(col, i)


D4=np.zeros((N,N))
col=np.zeros(N)
col[0]= (N**4/80. + N**2/12. -1./30)
col[1:]= 1./4*(-1)**(jlist)/np.sin(jlist*dx/2)**2*(-4./np.tan(jlist*dx/2)**2 -2./np.sin(jlist*dx/2)**2 + N**2)
for i in range(N):
  D4[:,i]=np.roll(col, i)





dataname='./../data/data_one.pkl'
data        = pickle.load( open( dataname, "rb" ) )
tlist       = data['tlist']#+dt          
ylist       = data['ylist'] 

f                   =   ylist[-100,:]
fx,fxx,fxxx,fxxxx   =   cal_der(f)


F       =   np.diagflat(f)
Fx      =   np.diagflat(fx)
Fxx     =   np.diagflat(fxx)
Fxxx    =   np.diagflat(fxxx)
Fxxxx   =   np.diagflat(fxxxx)

L0 = (-delta*alpha*Fxx +beta*Fxxx - Fxxxx)
L1 = (vff*np.eye(N) - 2*delta*alpha*Fx +beta*Fxx - Fxxx + 2*delta*gamma*Fxxx).dot(D1)
L2 = (-delta*alpha*(np.eye(N)+F) + beta*Fx + 4 *delta*gamma* Fxx).dot(D2)
L3 = (beta*(np.eye(N)+F) - Fx + 2 *delta*gamma* Fx).dot(D3)
L4 = -(np.eye(N)+F).dot(D4)

L=L0+L1+L2+L3+L4

Lambda,v=np.linalg.eig(L)
print(np.max(np.real(Lambda)))
print(np.where(np.real(Lambda)>0))
aaa=np.where(np.real(Lambda)>0)
print(Lambda[aaa])
# set_trace()
fig = plt.figure(0, figsize=(figwidth*1.4*2,figheight))
ax  = fig.add_subplot(121,alpha=1)#
plt.scatter(np.real(Lambda),np.imag(Lambda),fc='k',s=5)
ax.axvline(0,c='k',ls='--')

ax  = fig.add_subplot(122,alpha=1)#
plt.scatter(np.real(Lambda),np.imag(Lambda),fc='k',s=5)
ax.set_xlim([-100,20])
ax.set_ylim([-200,200])
ax.axvline(0,c='k',ls='--')
ax.text(-80,80,str(Lambda[aaa]))
set_trace()
# plt.plot(tlist,elist[:,0],c='tab:red',label=r'$\alpha$') # alpha

# plt.plot(tlist,elist[:,1],c='tab:green',label='surf') # surf
# plt.plot(tlist,elist[:,2],c='tab:red',alpha=0.7) # alpha-nl
# plt.plot(tlist,elist[:,3],c='tab:blue',label=r'$\beta$-nl',alpha=0.7) # beta-nl 
# plt.plot(tlist,elist[:,4],c='tab:green',alpha=0.7) # surf-nl
         
# plt.plot(tlist,np.sum(elist,axis=1),c='k')      
# plt.axhline(0,c='grey',ls='--')
# etot=np.sum(elist,axis=1)[-1]
# string='etot='+"{:e}".format(etot)
# ax.text(0.004,150,string)

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels,fontsize=gcafontSize,loc='best',numpoints=1) 

plt.xlabel(r"$\Re (\lambda)$",fontsize=1.0*gcafontSize)
plt.ylabel(r"$\Im (\lambda)$",fontsize=1.0*gcafontSize)
plt.tight_layout() 
fname = './../figures/eigenvalue.png'
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

