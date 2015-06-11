# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 23:15:09 2015

@author: rgrimson
"""
#%%
from scipy.stats import chi2
import stochastic_filters as sf

def compW(S,L,M,Filt,eta=0.97):
    N=2
    S1,S2=S
    l1,l2=L
    L1,L2=L
    m1,m2=M
    V=S.copy()
    H=S.copy()
    dg=2.0
    	
    if (Filt=='S'): # //Shannon
        for x in range(N):
            l=L[x]
            il=1.0/l
            s=S[x]
            gL=sf.Gamma(l)
            dgL=sf.digamma(l)
            pg1L=sf.polygamma(1,l)
            V[x]=pow((1-l)*pg1L+1-il,2)/(pg1L-il)+il
            H[x]=-log(l)+log(s)+l+(1-l)*dgL+log(gL)	
    
    if (Filt=='K'):  #Kullback-Leibler
        St=m2*m1*(L1-L2)/2.0/(m1+m2)* (log(S1/S2) - log(L1/L2) + sf.digamma(L1) - sf.digamma(L2)) + (L2*S1/S2 + L1*S2/S1)/2 - (L1+L2)/2;
    elif (Filt=='H'): #Hellinger
    	La=(L1+L2)/2
        St=4.0*m2*m1/(m1+m2)*(1 - sf.Gamma(La) * sqrt(pow(2/(L1*S2+L2*S1),L1+L2) * pow(L1*S2,L1) * pow(L2*S1,L2) / (sf.Gamma(L1) * sf.Gamma(L2)) ) );

    elif (Filt=='S'): #Shannon
    	v1=V[0]
    	h1=H[0]
    	v2=V[1]
    	h2=H[1]
    	w=1/(m1/v1+m2/v2)*(m1*h1/v1+m2*h2/v2)
    	St=m1*pow(h1-w,2)/v1+m2*pow(h2-w,2)/v2 #compute statistics
    	dg=1.0
    p=1-sf.chi2.cdf(St,dg) #compute p-value
    w=sf.p2w(p,eta)
    return St

#%%
mm,m1,mM=[0.1,0.7,1.0]
lm,l1,lM=[1.5,3.5,9.0]
eta=0.950
l=1000
S=array([m1, m1])
L=array([l1, l1])
M=array([9  , 9])

#VL=linspace(2.0,15.0,l)
#VS=repeat(3.5,len(VL))

VS=linspace(mm,mM,l)
VL=repeat(l1,l)

figure()
for Filt in ['K','H','S']:
    W=VL.copy()
    for i in range(l):
        L[0]=VL[i]
        S[0]=VS[i]
        W[i]=compW(S,L,M,Filt,eta=eta)
    #cla()    
    plot(VS,W)

VL=linspace(lm,lM,l)
VS=repeat(m1,len(VL))

figure()
for Filt in ['K','H','S']:
    W=VL.copy()
    for i in range(l):
        L[0]=VL[i]
        S[0]=VS[i]
        W[i]=compW(S,L,M,Filt,eta=eta)
    #cla()    
    plot(VL,W)

#%%
#Ph=sf.Phantom_Str(d=200)
Ph=sf.Phantom_circ(d=200)
PhC=sf.Speckle_Img(Ph,looks=3)
path='/home/rgrimson/Dropbox/Compartidas/Frery_SDNLM/Tests/Str'
#path='/home/rgrimson/Circ99_'
sf.Disp(PhC)
savefig(path+'_Corr.pdf')


for Filt in ['CK','CH','CS','DK','DH','DS']:
    PhF=sf.Filt(PhC,Filt,eta=0.99)
    sf.Disp(PhF)
    savefig(path+'_'+Filt+'.pdf')


#%%
import stochastic_filters as sf

Ph=sf.Phantom_circ(d=100)
PhC=sf.Speckle_Img(Ph,looks=3)
PhF=sf.Filt(PhC,'CH',eta=0.99)