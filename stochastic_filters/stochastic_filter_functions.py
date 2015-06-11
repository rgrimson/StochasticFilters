# -*- coding: utf-8 -*-
#from __future__ import division

"""
Created on Mon Dec 15 19:04:54 2014

@author: rgrimson
"""

#%%
import matplotlib.pyplot as plt
import stochastic_filters as sf

import numpy as np
#from numpy import *
#import scipy as sp
import gdal
#import sys
#from gdalconst import *
import os
import os.path
#from scipy import ndimage

#from scipy.stats import tvar
#from scipy.stats import norm

#from scipy.special import gamma as Gamma
from scipy.stats import gamma #, chi2
from scipy.special import polygamma
from scipy.special import digamma
from scipy import percentile
#from time import localtime

#import matplotlib.pyplot as plt

#Ftype=dtype([('Filter', np.str_, 1), ('Decentered', np.bool, 1),('Sw', np.int16, 1),('eta',np.float32,1)])

#Draws a phantom with an S
def Phantom_sin(d=500):
    M = np.ones([d,d])
    for y in range(d):
        for x in range(d):
            if np.sin((x+y)*np.pi/d)>=(y-x)*np.pi/d:
                M[y,x]*=2
    return M

#Draws a phantom with a circle
def Phantom_circ(d=500,r=0.66):
    m=(d-1)/2
    r*=m
    r2=r*r
    M = np.ones([d,d])
    for y in range(d):
        for x in range(d):
            if (x-m)**2+(y-m)**2<r2:
                M[y,x]*=5
    return M

#Draws a phantom with three regions. d is de diameter.
def Phantom_Y(d=500):
    m=int((d-1)/2)
    mx=int((2*d-1)/3)
    M = np.ones([d,d])*0.9
    for y in range(d):
        for x in range(mx):
            if (x-mx)<2*(y-m):
                M[y,x]*=0.3
        for x in np.arange(mx,d):
            if 3*(x-mx)>=-2*(y-m):
                M[y,x]*=0.6
    return M

def Phantom_4SQ(d=500):
#Draws a phantom with four squares
    r=int(d/5)
    r4=int(d/20)
    mid=int(d/2)
    count = np.arange(r4,5*r-r4,23) # Counter for the white spots

    squares = np.ones([d,d])*10 #mean = 10

    # Each square has a different mean
    squares[r+1:2*r+1,1*r+1:2*r+1]=02 # Mean = 2
    squares[r+1:2*r+1,3*r+1:4*r+1]=40 # Mean = 40
    squares[3*r+1:4*r+1,1*r+1:2*r+1]=60# Mean = 60
    squares[3*r+1:4*r+1,3*r+1:4*r+1]=80 # Mean = 80

    # Vertical squares 4x2
    squares[count,   mid-1:mid+1]  =245 # Mean = 245
    squares[count+1, mid-1:mid+1]=245 # Mean = 245
    squares[count+3, mid-1:mid+1]=245 # Mean = 245

    # Horizontal squares 4x4
    squares[mid-2:mid+2, count]   =245 # Mean = 245
    squares[mid-2:mid+2, count+1] =245 # Mean = 245
    squares[mid-2:mid+2, count+2] =245 # Mean = 245
    squares[mid-2:mid+2, count+3] =245 # Mean = 245
    return squares

#Draws a phantom with stripes
def Phantom_Str(d=500):
    """Class methods are similar to regular functions.
    
    Note:
      Do not include the `self` parameter in the ``Args`` section.
    
    Args:
      param1: The first parameter.
      param2: The second parameter.
    
    Returns:
      True if successful, False otherwise.
    
    """

    #n =int((sqrt(24*d+1.0)+1)/12.0)
    Ph = np.ones([d,d])*1 #mean = 10
    if d<60: return Ph
        
    n = int((np.sqrt(16*d-87)-3)/8.0)
    
    M=(4*n-3)
    r=6
    count = np.arange(r,r+n*(M+r),M+r)
    count=count+int((d-M-count[-1]+count[0])/2.0-count[0])
    
    for i in range(n):
      for g in range(4*i+1):
        Ph[count[i]+g,r:d-45]=4
      Ph[count[i]:count[i]+1,d-35]=4
      Ph[count[i]:count[i]+5,d-25:d-25+5]=4
      Ph[count[i]:count[i]+9,d-15:d-15+9]=4
    return Ph

#adds speckle noise to the image
def Speckle_Img(Img,looks=1):
    dimy,dimx=Img.shape
    Speckle = gamma.rvs(looks,scale=1.0/looks,size=dimx*dimy).reshape(dimy,dimx)
    return Img*Speckle

#shows an image
def Disp(Img,vmin=0,vmax=0,fname=""): 
    if (vmin==vmax):
        vmin=percentile(Img,2)
        vmax=percentile(Img,98)
    plt.imshow(Img,vmin=vmin,vmax=vmax,cmap = plt.get_cmap('gray'),interpolation='None')
    plt.axis('off')
    if (fname!=""):
        plt.savefig(fname,bbox_inches='tight')

#solves log(x)-digamma(x)=y using Newton method
def NR_L(y):
    def func(x):
        return np.log(x)-digamma(x)

    def deriv(x):
        return 1/x-polygamma(1,x)

    eps=0.0001
    if (y<eps): 
        return NR_L(eps)
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
#def Wishartpdf(x,L,s):
#    L=min(120,L)
#    return gamma.pdf(x,L,scale=s/L)

#computes the maximum likelyhood Whishart parameters for a given sample
def MLtheta(W):
    m=np.mean(W)
    return m, NR_L(np.log(m)-sum(np.log(W))/np.prod(W.shape)) # mean over the windows = ML sigma. Solve Newton-Raphson = ML number of looks

#save image in ENVI format (use geocoding from src image if given)
def Save_ENVI(dst_filename,data, src_filename = None):
    format = "ENVI"
    driver = gdal.GetDriverByName( format )
    if (src_filename == None):
        dst_ds = driver.Create(dst_filename, data.shape[1],data.shape[0],1,gdal.GDT_Float64)
    else:
        dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
        dst_ds = driver.CreateCopy( dst_filename, dataset, 0 )
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds  = None

#save image in GeoTiff format. Use geocoding from src image
def Save_GTIFF(src_filename, dst_filename,data):
    format = "GTIFF"
    driver = gdal.GetDriverByName( format )
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    dst_ds = driver.CreateCopy( dst_filename, dataset, 0 )
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds  = None
    
#save image stack in GeoTiff format. Use geocoding from src image
def Save_Stk_GTIFF(src_filename, dst_filename,data):
    format = "GTIFF"
    driver = gdal.GetDriverByName( format )
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    dst_ds = driver.CreateCopy( dst_filename, dataset, 0 )
    for i in range(data.shape[0]):
        dst_ds.GetRasterBand(i+1).WriteArray(data[i])
    dst_ds  = None

#load image stack 
def loadStack(src_filename):
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    tifArray = dataset.ReadAsArray()
    return tifArray
    #return np.array(tifArray,dtype=float64)

#Filter each image in a stack with given parameters
def FiltStack(Stk, Filt='CK', Er=1, Sr=2, eta=0.95):
    d=Stk.shape[0]
    F=Stk.copy()
    for i in range(d):
        print ("Filtering Band:",i)
        F[i,:,:]=sf.Filt(Stk[i,:,:],Filt, Er=Er, Sr=Sr, eta=eta)
    return F


#transforms a p-value into a weight for the convolution kernel
def p2w(p,eta):
    if p>=eta:
        return 1
    elif p<(eta/2):
        return 0
    return (p-eta/2)*2/eta

#read a band from an image file
def loadBand(src_filename, band=0):
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    tifArray = dataset.ReadAsArray()

    multiband=(len((tifArray.shape))==3)
    if (multiband):
            return tifArray[band,:,:]
    else:
        return tifArray

#compute Q-index between images
def Q(O,F):
    Ov=O.var()
    Om=O.mean()
    Fv=F.var()
    Fm=F.mean()
    OFd=1.0/np.prod(O.shape)*((O-Om)*(F-Fm)).sum()
    return OFd*4.0*Om*Fm/((pow(Om,2)+pow(Fm,2))*(Ov+Fv))

#computes mean and n.looks for the slices of Img given in Eval_Regions
def ComputeImgStats(Img,Eval_Regions=[]):
    l=len(Eval_Regions)
    Rmean=np.zeros([l])
    Renl=np.zeros([l])
    for i in range(l):
        r=Eval_Regions[i]
        #m=(r[0].stop-r[0].start)*(r[1].stop-r[1].start)
        Rmean[i]=Img[r].mean()
        Renl[i]=NR_L(np.log(Rmean[i])-np.log(Img[r]).mean())
    return Renl,Rmean

#given a phantom with n different values computes a similar matrix of integers (classification).
def ClassifyPhantom(Ph):
    C=np.ones(Ph.shape,dtype=int)*0
    V=() #tuple fo values in Ph
    K=0  #number of classes
    for y in range(Ph.shape[0]):
        for x in range(Ph.shape[1]): #add a class?#
            if Ph[y,x] not in V:
                V=V+(Ph[y,x],)
                K+=1
    V=np.sort(V) #sort classes
    #define classification matrix
    for y in range(Ph.shape[0]):
        for x in range(Ph.shape[1]):
            for i in range(V.shape[0]):
                if Ph[y,x]==V[i]:
                    C[y,x]=i
    return C, K

#Classify an image into K classes using EM for GMM from ML parameters
def Classify(Img,mode="C",K=2,verbose=True):
    S=np.zeros(Img.shape)
    L=np.zeros(Img.shape)
    m=np.zeros(Img.shape)
    dimy, dimx=Img.shape
    
    if (mode[0]=="D"):
        sf.Compute_ML_Param_NMWin( Img, S, L, m, dimx, dimy)
    else:
        sf.Compute_ML_Param( Img, S, L, m, 1, dimx, dimy)
    S=S.reshape(np.prod(S.shape))
    L=L.reshape(np.prod(L.shape))
    d=array([S,log(L)]).T
    import pypr.clustering.gmm as gmm
    p_k=0
    while np.prod(p_k)==0:
        print "Computing ML clusters using Gaussian Mixture EM algorithm"
        cen_lst, cov_lst, p_k, logL = gmm.em_gm(d, K = K ,max_iter =2000,verbose=verbose)

    print "Log likelihood = ", logL
    C=gmm.gm_assign_to_cluster(d, cen_lst, cov_lst, p_k)
    C=C.reshape(Img.shape)
    #sort classes so that their means follow an ascending order
    #    x = zeros(K, dtype={'class':(int,32), 'value':(float,1)})
    #    x[:]['class']=range(K)
    #    for i in range(K):
    #        x[i]['value']=Img[C==i].mean()
    #    x=sort(x,order='value')
    #    C=x[C]['class']
    return C


#Classify an image into K classes using EM for GMM from ML parameters
#def Classify_fromML(Img,S,L,m,K=2,verbose=True):
#    S=S.reshape(np.prod(S.shape))
#    L=L.reshape(np.prod(L.shape))
#    d=np.array([S,np.log(L)]).T
#    import pypr.clustering.gmm as gmm
#    p_k=0
#    while np.prod(p_k)==0:
#        print "Computing ML clusters using Gaussian Mixture EM algorithm"
#        cen_lst, cov_lst, p_k, logL = gmm.em_gm(d, K = K ,max_iter =2000,verbose=verbose)
#    print "Log likelihood = ", logL
#    C=gmm.gm_assign_to_cluster(d, cen_lst, cov_lst, p_k)
#    C=C.reshape(Img.shape)
#    return C

#refine a vector to finner plot
def refine(x):
    l=len(x)
    z=np.zeros(2*l-1)
    for i in range(l-1):
        z[2*i]=x[i]
        z[2*i+1]=(x[i]+x[i+1])/2
    z[2*l-2]=x[l-1]
    return z

#Compute the kappa statistical measure of inter-rater agreement from confussion matrix
def kappa(M):
    dim=M.shape[0]
    X=np.zeros([dim])
    Y=np.zeros([dim])
    D=np.zeros([dim])
    for i in range(dim):
        X[i]=M[:,i].sum()
        Y[i]=M[i,:].sum()
        D[i]=M[i,i]
    t=X.sum()
    PrA=D.sum()/t
    PrE=(X*Y).sum()/(t*t)
    return (PrA-PrE)/(1-PrE)

#Compute the kappa statistical measure of inter-rater agreement between two images
def kappa2(M,N):
    K=max(M.max(),N.max())+1
    C=np.zeros([K,K])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            C[M[i,j],N[i,j]]+=1
    for i in range(K-1):
        M=(C[0:K-i,0:K-i]).max()
        for j in range(K-i):
            m=max(C[j,0:K-i])
            n=max(C[0:K-i,j])
            if m==M: r=j
            if n==M: c=j
        C[:,[K-i-1,c]]=C[:,[c,K-i-1]]
        C[[K-i-1,r],:]=C[[r,K-i-1],:]
    return kappa(C)

#rename file with a new name
def BackUpExisting(filename):
    if os.path.isfile(filename):
            i=0
            fname=filename  + str(i)
            while os.path.isfile(fname):
                i+=1
                fname=filename  + str(i)
            os.rename(filename, fname)

#create an image showing the evaluation regions
def DrawEvalRegions(Img,Eval_Regions):
    #draw image with evaluation regions distingushed
    M=percentile(Img,2)+percentile(Img,98)/2
    Imgc=Img.copy()
    for r in Eval_Regions:
        Imgc[r[0].start,r[1]]=M
        Imgc[r[0].stop-1,r[1]]=M
        Imgc[r[0],r[1].start]=M
        Imgc[r[0],r[1].stop-1]=M
    return Imgc

#evaluate a filter. Compute Q-index and, given evaluation regions, radiometric preservation and number of looks for each region
def FilterEval(I_Filt,I_Orig,Eval_Regions=[]):
    l=len(Eval_Regions)
    result=np.zeros([2*l+1])
    # save evalution regions
    if l>0:
      for iR in range(len(Eval_Regions)):
        r=Eval_Regions[iR]
        Omean=I_Orig[r].mean()
        Fmean=I_Filt[r].mean()
        #m=(r[0].stop-r[0].start)*(r[1].stop-r[1].start)
        Fenl=NR_L(np.log(Fmean)-np.log(I_Filt[r]).mean())
        result[2*iR+1] = Fmean/Omean
        result[2*iR+2] = Fenl
    result[0]=Q(I_Filt,I_Orig)
    return result

#evaluate a tuple of filters as in FilterEval
def FiltersEval(I_Corr,I_Orig=None,n_iter=1,V_Sr=[2],V_eta=[0.97],Filters=["CH","CK","CS"],Eval_Regions=[],saveImg=False,srcImg = None,info_str=''):
    #initialize
    S=np.zeros(I_Corr.shape,dtype='float64')
    L=np.zeros(I_Corr.shape,dtype='float64')
    m=np.zeros(I_Corr.shape,dtype='float64')
    print "Evaluating Result: " + info_str
    n_Filters=len(Filters)
    n_R=len(Eval_Regions)
    n_eta=len(V_eta)
    n_Sr=len(V_Sr)
    #if I_Orig!=None:
    #    k=1
    #    PhC, K=ClassifyPhantom(I_Orig)
    #else:
    #    k=0
    k=0
    results=np.zeros([n_Filters,n_eta,n_Sr,n_iter,2*n_R+1+k])
    #preform all filters
    for iFilt in range(n_Filters):
        Filt_act = Filters[iFilt]
        for ieta in range(n_eta):
            eta=V_eta[ieta]
            for iSr in range(n_Sr):
                Sr=V_Sr[iSr]
                I_Filt=I_Corr.copy()
                for iIt in range(n_iter):
                    inf_str= info_str + "_"+ Filt_act + '_It' + str(iIt+1) + "_Sw" + str(2*Sr+1) + "_e" + str(int(100*eta))
                    print inf_str
                    I_Act=I_Filt.copy()
                    #print "aca"
                    if (Filt_act[0]=='D'):
                        sf.Compute_ML_Param_NMWin(I_Act,S,L,m,I_Act.shape[1],I_Act.shape[0])
                    else:
                        sf.Compute_ML_Param(I_Act,S,L,m,1,I_Act.shape[1],I_Act.shape[0])
                    print inf_str
                    #print "aqui1"
                    sf.Filter_fromMLParam(I_Act, I_Filt, S,L,m,I_Act.shape[1],I_Act.shape[0],Sr,eta,Filt_act[1])
                    #print "aqui2"
                    if (I_Orig==None):
                        results[iFilt,ieta,iSr,iIt]=FilterEval(I_Filt,I_Corr,Eval_Regions=Eval_Regions)
                    else:
                        results[iFilt,ieta,iSr,iIt,0:2*n_R+1]=FilterEval(I_Filt,I_Orig,Eval_Regions=Eval_Regions)
                        #I_Class=Classify_fromML(I_Filt,S,L,m,K=K)
                        #results[iFilt,ieta,iSr,iIt,2*n_R+1]=kappa2(PhC,I_Class)
                    #Save image
                    if saveImg:
                        print "Saving: " + inf_str
                        Save_ENVI(inf_str,I_Filt,src_filename = srcImg)
    return results

#Evaluate different filters as in FilterEval using Monte Carlo method.
def MC_FiltersEval(I_Orig,n_mc_iter=100,looks=3,n_iter=1,V_Sr=[2],V_eta=[0.97],Filters=['CH','CK','CS'],Eval_Regions=[],base_fname='MCFE'):
    n_Filters=len(Filters)
    n_R=len(Eval_Regions)
    n_eta=len(V_eta)
    n_Sr=len(V_Sr)
    results=np.zeros([n_mc_iter,n_Filters,n_eta,n_Sr,n_iter,2*n_R+1])
    #SaveEvalRegions(I_Orig,orig_fname,Eval_Regions)
    for i_mc in range(n_mc_iter):
        I_Corr=Speckle_Img(I_Orig,looks=looks)
        results[i_mc]=FiltersEval(I_Corr,I_Orig=I_Orig,n_iter=n_iter,V_Sr=V_Sr,V_eta=V_eta,Filters=Filters,Eval_Regions=Eval_Regions,info_str=base_fname+'_MC'+str(i_mc),saveImg=False)
    return results
    
#Filter an image with given Filter (CH,DH,CK,DK,CS,DS), estimation radius , search radius, and eta value
def Filt(Img, F,Er=1,Sr=2,eta=0.97):
    I_Filt=np.zeros(Img.shape,dtype='float64')
    if F[0]=='D':
        Er=0
    sf.Filter(Img, I_Filt,Img.shape[1],Img.shape[0],Er,Sr,eta,F[1])
    return I_Filt
    
