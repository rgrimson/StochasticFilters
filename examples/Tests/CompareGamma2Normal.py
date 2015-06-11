# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 13:31:37 2015

@author: rgrimson
"""
from scipy.special import gamma as Gamma
from scipy.special import polygamma
from scipy.special import digamma
from scipy.stats   import gamma
from scipy.stats import norm


def NR_L(y):
    def func(x):
        return np.log(x)-digamma(x)

    def deriv(x):
        return 1/x-polygamma(1,x)

    eps=0.000001
    if (y<eps): return 1/eps
    #start on the left of the root
    x=2.0
    f=func(x)
    while (f>y):
        x=x*2
        f=func(x)
    while (f<y):
         x=x/2
         f=func(x)
    #perform Newton Raphson
    #print x,f
    while (np.abs(f-y)>eps):
        der=deriv(x)
        x=x+(y-f)/der
        f=func(x)
    return x

#computes Whishart pdf
def Wishartpdf(x,L,s):
    #L=min(120,L)
    return gamma.pdf(x,L,scale=s/L)

#computes the maximum likelyhood Whishart parameters for a given sample
def MLtheta(W):
    m=mean(W)
    return m, NR_L(log(m)-sum(log(W))/prod(W.shape)) # mean over the windows = ML sigma. Solve Newton-Raphson = ML number of looks
    
#%%
N=50 #number of pixels per segment
cla()
for N in [2,5,10,15,20,25,30,35,40,45,50,100]:#,11,12,13,14,15,16,17,18,19,20]:
    K=1000 #number of segments

    M=zeros(K)
    S=zeros(K)
    m=0
    s=1
    for k in range(K):
        V=norm(m,scale=s).rvs(N)
        M[k]=V.mean()
        S[k]=V.std()
    
    #cla()
    #hist(S,bins=100,normed=True) 
    x=linspace(S.mean()-4*S.std(),S.mean()+4*S.std(),250)
    z=norm(S.mean(),scale=S.std()).pdf(x) 
    y=Wishartpdf(x,MLtheta(S)[1],S.mean())  
    plot(x,y,'r')
    plot(x,z,'b')
        
    m,L=MLtheta(S)
    print S.mean(),S.std()
    k=L
    t=m/L
    s=sqrt(k)*t
    print L, s

#%%
#%%
K=1000 #number of segments
N=50 #number of pixels per segment
M=zeros(K)
S=zeros(K)
m=0
s=1
for k in range(K):
    V=norm(m,scale=s).rvs(N)
    M[k]=V.mean()
    S[k]=V.std()

cla()
hist(S,bins=100,normed=True) 
x=linspace(S.mean()-4*S.std(),S.mean()+4*S.std(),250)
z=norm(S.mean(),scale=S.std()).pdf(x) 
y=Wishartpdf(x,MLtheta(S)[1],S.mean())  
plot(x,y,'r')
plot(x,z,'b')
    
m,L=MLtheta(S)
print S.mean(),S.std()
k=L
t=m/L
s=sqrt(k)*t
print L, s