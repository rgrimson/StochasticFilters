# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 20:08:33 2015

@author: rgrimson
"""
# PYTHON 4
import stochastic_filters as sf
#import numpy as np


import time;


n_mc_iter=100
Filters=['CH','CK','CS','DH','DK','DS']
V_eta=[0.9,0.95,0.99]
V_Sr=[2,3,5]
n_iter=3

#n_mc_iter=2
#n_iter=2
#V_Sr=[2,3]
#Filters=['CH','DH']
#V_eta=[0.99]


d=180
Ph=sf.Phantom_circ(d=d)
Eval_Regions=[[slice(0,d/7,1),slice(0,d,1)],[slice(d/3,2*d/3,1),slice(d/3,2*d/3,1)]]
PhER=sf.DrawEvalRegions(Ph,Eval_Regions)
#sf.Disp(PhER)

localtime = time.localtime(time.time())
resultsCirc180=sf.MC_FiltersEval(Ph,n_mc_iter=n_mc_iter,looks=3,n_iter=n_iter,V_Sr=V_Sr,V_eta=V_eta,Filters=Filters,Eval_Regions=Eval_Regions,base_fname='MC_Circ180')
resultsCirc180.tofile('resultsCirc180_.out')

print "Start time :", localtime
localtime = time.localtime(time.time())
print "Finish time :", localtime
#print "Result is OK:", not(isnan(resultsCirc180).any())
#isnan(resultsCirc180).any()
#isnan(resultsCirc180).any()
#%%
#PYTHON 17 y 20
import stochastic_filters as sf
#import numpy as np
import time;


n_mc_iter=100
Filters=['CH','CK','CS','DH','DK','DS']
V_eta=[0.9,0.95,0.99]
V_Sr=[2,3,5]
n_iter=3


d=180
Ph=sf.Phantom_Str(d=d)
Eval_Regions=[[slice(148,167,1),slice(10,130,1)],[slice(16,35,1),slice(10,130,1)]]
PhER=sf.DrawEvalRegions(Ph,Eval_Regions)
#sf.Disp(PhER)
resultsStr=sf.MC_FiltersEval(Ph,n_mc_iter=n_mc_iter,looks=3,n_iter=n_iter,V_Sr=V_Sr,V_eta=V_eta,Filters=Filters,Eval_Regions=Eval_Regions,base_fname='MC_Str180')
resultsStr.tofile('resultsStr180.out')

print "Start time :", localtime
localtime = time.localtime(time.time())
print "Finish time :", localtime

#%%
# PYTHON 16 y 19
import stochastic_filters as sf
#import numpy as np
import time;


n_mc_iter=100
Filters=['CH','CK','CS','DH','DK','DS']
V_eta=[0.9,0.95,0.99]
V_Sr=[2,3,5]
n_iter=3


r=100
d=5*r
m=3
Ph=sf.Phantom_4SQ(d=d)
Eval_Regions=[[slice(r+m+1,2*r-m+1,1),slice(r+m+1,2*r-m+1,1)],[slice(r+m+1,2*r-m+1,1),slice(3*r+m+1,4*r-m+1,1)],[slice(3*r+m+1,4*r-m+1,1),slice(r+m+1,2*r-m+1,1)],[slice(3*r+m+1,4*r-m+1,1),slice(3*r+m+1,4*r-m+1,1)],[slice(0,d/7,1),slice(0,d/2-m,1)]]
PhER=sf.DrawEvalRegions(Ph,Eval_Regions)
#sf.Disp(PhER)
results4SQ=sf.MC_FiltersEval(Ph,n_mc_iter=n_mc_iter,looks=3,n_iter=n_iter,V_Sr=V_Sr,V_eta=V_eta,Filters=Filters,Eval_Regions=Eval_Regions,base_fname='MC_4SQ500')
results4SQ.tofile('results4SQ__.out')

print "Start time :", localtime
localtime = time.localtime(time.time())
print "Finish time :", localtime
