import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import sys
from pdb import set_trace
from scipy.fftpack import fft, ifft
import pickle
import parameters as para

figheight      = 6
figwidth       = 6
lineWidth      = 2.
textFontSize   = 18
gcafontSize    = 12

print('######################################')
print('Reading parameters...')
# physical parameters
alpha   = para.alpha
gamma   = para.gamma
beta    = para.beta
delta   = para.delta

# simulation parameters
N       = para.N
x       = para.x
dx      = x[2]-x[1]
wavek   = para.wavek

Ttot    = para.Ttot
h       = para.dt
Data_output=para.Data_output
Plot_output=para.Plot_output

u0      = para.u0

print('alpha,beta,gamma,delta=',alpha,beta,gamma,delta)
print('N=',N)
print('dt=',h)
print('######################################')

if (para.Restart==0)&(para.Perturbation ==0):
    t       = 0.0
    Resi    = 0
    u       = u0
    NT      = int(Ttot/h)
    dataj   = 1
    print('Initialization from t=0, no perturbation')

if (para.Perturbation ==1):
    t       = 0.0
    Resi    = 0
    data    = pickle.load( open( para.Basename, "rb" ) )      
    u0      = data['y']
    up      = para.up
    u       = u0+up
    NT      = int(Ttot/h)
    dataj   = 1
    print('Initialization from t=0, with perturbation')

if (para.Restart==1):
    ET          = para.Extratime    
    data        = pickle.load( open( para.Resname, "rb" ) )    
    t           = data['time']
    u           = data['y']
    NT          = int(ET/h)+1
    # set_trace()
    dataj       = para.Res_i
    print('Initialization from restart file at t=',t)


v       = fft(u)
v0=v*1.0
pkl_name='./data/data_'+str(0)+'.pkl'
data={}
data['time']  = t   
data['y']     = u0
data['fft']   = v0
output = open(pkl_name, 'wb')
pickle.dump(data, output)
output.close()
print (pkl_name," saved!")



## Linear part in fourier space
L = -delta*alpha*(wavek**2) - wavek**4 - beta*1j*wavek**3
Nonl =  1j*wavek**3 - beta*wavek**2
## Fourier mulitplier
E = np.exp(h * L)
E2 = np.exp(h * L / 2.0)

## Number of points on the circle
M = 64

## Choose radius 1
r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
r = r.reshape(1, -1)
r_on_circle = np.repeat(r, N, axis=0)

## Define LR
LR = h * L

LR = LR.reshape(-1, 1)
LR = np.repeat(LR, M, axis=1)

LR = LR + r_on_circle


# averaged weights
Q = h*np.mean( (np.exp(LR/2.0)-1)/LR, axis=1 )

f1 = h*np.mean( (-4.0-LR + np.exp(LR)*(4.0-3.0*LR+LR**2))/LR**3, axis=1 )

f2 = h*np.mean( (2.0+LR + np.exp(LR)*(-2.0 + LR))/LR**3, axis=1 )

f3 = h*np.mean( (-4.0-3.0*LR - LR**2 + np.exp(LR)*(4.0 - LR))/LR**3, axis=1 )

def cal_nonlinear(y_hat):
    y   = np.real(ifft(y_hat))
    N1  = Nonl*y_hat
    yna = np.real(ifft(N1))*y #physical domain
    Ya  = 1j * wavek * fft(yna)

    Yx = 1j * wavek *y_hat
    yx = np.real(ifft(Yx))
    ynb = alpha/2*y**2 + gamma*yx**2 #physical domain
    Yb  = (-wavek**2*fft(ynb))*delta
    g=Ya+Yb

    return g

#===========================================================================
## main loop time stepping
for i in range(1,NT+1):

    t=i*h


    Nv = cal_nonlinear(v)
    
    a = E2 * v + Q * Nv

    Na = cal_nonlinear(a)
    
    b = E2 * v + Q * Na

    Nb = cal_nonlinear(b)
    
    c = E2 * a + Q * (2.0*Nb - Nv)

    Nc = cal_nonlinear(c)
    
    v = E*v + Nv*f1 + 2.0*(Na + Nb)*f2 + Nc*f3

    if (i%Data_output==0):

        y=np.real(ifft(v))

        print ("==========================================")
        print ("iteration:",i,'/',NT)
        print ("time:",t)      
        print ('mean,max:',np.mean(y),np.max(y))  

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (i%Plot_output==0):
            fig = plt.figure(0, figsize=(figwidth*1.2,figheight*0.25))
            ax  = fig.add_subplot(111,alpha=1)#
            ax.plot(x,y,c='k')
            ax.set_xlabel('x',fontsize=0.8*gcafontSize)
            ax.set_ylabel('y',fontsize=0.8*gcafontSize)
            ax.set_ylim([-1.0,1.0])
            string='t='+str(round(t,4))
            ax.text(5.5,0.2,string,fontsize=0.8*gcafontSize)

            figname="frame_"+str(dataj)+'.png'
            plt.tight_layout()    
            plt.savefig('./movie/'+figname)
            plt.close()
            print (figname," saved!")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        pkl_name='./data/data_'+str(dataj)+'.pkl'
        data={}
        data['time']  = t   
        data['y']     = y
        data['fft']   = fft(y)
        output = open(pkl_name, 'wb')
        pickle.dump(data, output)
        output.close()
        print (pkl_name," saved!")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        dataj=dataj+1





