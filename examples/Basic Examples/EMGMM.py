# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:58:43 2015

@author: rgrimson
"""

from scipy.stats import norm
K=2

#loc=[0,1,4,6]
#sca=[.2,.3,.2,.1]
#siz=[500,300,600,200]

loc=random(K)*K
sca=random(K)
siz=randint(500,size=K)

X_Orig=[normal(loc=loc[k],scale=sca[k],size=siz[k]) for k in range(K)]
X=concatenate(X_Orig)
     
mu=X.mean()
sigma=X.std()
X-=mu
X/=sigma
X_Scl=X_Orig
for k in range(K):
      X_Scl[k]-=mu
      X_Scl[k]/=sigma      


N=len(X)          
Z=np.random.randint(K, size=N)
dx=(X.max()-X.min())/(K-1)
m=random(K)#percentile(X,[5,30,70,95])
#m=arange(X.min(),X.max()+dx,step=dx)
s=repeat(dx/K,K)
r=zeros([N,K])
w=repeat(1.0/K,K)


def P(bins=250):
    global X
    U=linspace(X.min(),X.max(),1000) 
    cla()
    V,W,Y=hist(X_Scl,bins=bins,normed=True) 
    dx=W[1]-W[0]
    PDF=zeros(U.shape)
    for k in range(K):
        plot(U,w[k]*norm(m[k],sqrt(s[k])).pdf(U)*K)
        PDF+=w[k]*norm(m[k],sqrt(s[k])).pdf(U)*K
    plot(U,PDF)
        
#
def E_step(): 
    global r,R
    for i in range(N):    
        for k in range(K):    
            r[i,k]=w[k]*norm(m[k],sqrt(s[k])).pdf(X[i])
        r[i,:]/=r[i,:].sum()
    R=r.sum(axis=0)

def M_step():
    global w,m,s
    w=R/N
    for k in range(K):
        m[k]=(r[:,k]*X).sum()/R[k]
        s[k]=(r[:,k]*(X-m[k])**2).sum()/R[k]#(r[:,k]*X*X).sum()/R[k]-m[k]*m[k]
        
def S():
    E_step()
    M_step()
        
P()    
#S();P()
    
                        
        
    