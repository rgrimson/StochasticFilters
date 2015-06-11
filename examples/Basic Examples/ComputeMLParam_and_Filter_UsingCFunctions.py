#!/usr/bin/python2.7
import stochastic_filters as sf
import numpy as np

#f = np.float64(1.0)
#i = np.int32(1.0)

#print sf.digamma(f), digamma(1.0)
#print sf.polygamma(i,f), polygamma(1,1.0)

#print sf.NR_L(1.0)

Ph=sf.Speckle_Img(sf.Phantom_circ(d=50),looks=4)

S_=Ph.copy()
L_=Ph.copy()
m_=Ph.copy()
I_=Ph.copy()
i =Ph.copy()

#sf.Compute_ML_Theta_(Ph,S_,L_,m_,1,Ph.shape[0],Ph.shape[1])

sf.Compute_ML_Win_(Ph,S_,L_,m_,Ph.shape[0],Ph.shape[1])
sf.Filter_fromMLParam_(Ph,S_,L_,m_,I_,Ph.shape[0],Ph.shape[1],2,0.97,'K')

s,l,m = sf.Compute_ML_Theta_Decentered(Ph)
sf.Filter_fromMLParam_KL_(Ph,s,l,m,i,Ph.shape[0],Ph.shape[1],2,0.97)

figure()
sf.Disp(i)
figure()
sf.Disp(I_)

#
#print (s-S_).max()
#print (l/L_).max()
#print (l).max()
#print (m-m_).max()
#print s
#print 's'
#print S_
#print 'S_'
#print l
#print 'l'
#print L_
#print 'L_'
#print m
#print 'm'
#print m_
#print 'm_'
###################################
#TIMEIT
#
#import timeit
#
#setup = '''import stochastic_filters as sf
#Ph=sf.Speckle_Img(sf.Phantom_circ(d=50),looks=4)
#S_=Ph.copy()
#L_=Ph.copy()
#m_=Ph.copy()
#I_=Ph.copy()
#i =Ph.copy()
#'''
#
#
##if False:
#
#np  = 10
#
##comand = 'sf.Compute_ML_Theta_(Ph,S_,L_,m_,1,Ph.shape[0],Ph.shape[1])'
##t1 = timeit.timeit(comand, setup=setup, number = np)
#
##comand = 's,l,m = sf.Compute_ML_Theta(Ph)'
##t2 = timeit.timeit(comand, setup=setup, number = np)
#
##print t1/np,t2/np,t1/t2
#
#
#
#
#comand = 'sf.Compute_ML_Win_(Ph,S_,L_,m_,Ph.shape[0],Ph.shape[1]);sf.Filter_fromMLParam_KL_(Ph,S_,L_,m_,I_,Ph.shape[0],Ph.shape[1],2,0.97)'
#t1 = timeit.timeit(comand, setup=setup, number = np)
##comand = 'sf.Filter_fromMLParam_KL_(Ph,S_,L_,m_,I_,Ph.shape[0],Ph.shape[1],2,0.97)'
##t12 = timeit.timeit(comand, setup=setup, number = np)
#
#comand = 's,l,m = sf.Compute_ML_Theta_Decentered(Ph);i=sf.SDNLMfilter_fromMLParam_KL(Ph,s,l,m,Sr=2,eta=0.97)'
#t2 = timeit.timeit(comand, setup=setup, number = np)
##comand = 'sf.Filter_fromMLParam_KL_(Ph,s,l,m,i,Ph.shape[0],Ph.shape[1],2,0.97)'
##t22 = timeit.timeit(comand, setup=setup, number = np)
#
##t1=t11+t12
##t2=t21+t22
#
#print t1/np,t2/np,t2/t1
#
#
