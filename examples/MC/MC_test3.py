# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 20:08:33 2015

@author: rgrimson
"""
#%%

import stochastic_filters as sf
#import numpy as np
import time;
import struct;


n_mc_iter=100
n_iter=3
V_Sr=[2,3,5]
Filters=['CH','CK','CS','DH','DK','DS']
V_eta=[0.9,0.95,0.99]

#n_mc_iter=5
#n_iter=1
#V_Sr=[2]
#Filters=['CH','DH']
#V_eta=[0.99]

localtime = time.localtime(time.time())
print "Local current time :", localtime

d=180
Ph=sf.Phantom_Str(d=d)
Eval_Regions=[[slice(148,167,1),slice(10,130,1)],[slice(16,35,1),slice(10,130,1)]]
PhER=sf.DrawEvalRegions(Ph,Eval_Regions)
#sf.Disp(PhER)

#%%
resultsStr=sf.MC_FiltersEval(Ph,n_mc_iter=n_mc_iter,looks=3,n_iter=n_iter,V_Sr=V_Sr,V_eta=V_eta,Filters=Filters,Eval_Regions=Eval_Regions,base_fname='MC_Str500')
resultsStr.tofile('resultsStr180_.out')
#%%

print "Start time :", localtime
localtime = time.localtime(time.time())
print "Finish time :", localtime


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%
import stochastic_filters as sf
#import numpy as np
import time;
import struct;
import threading

#%% Define Phantom

d=100
Ph=sf.Phantom_circ(d=d)
Eval_Regions=[[slice(0,d/7,1),slice(0,d,1)],[slice(d/3,2*d/3,1),slice(d/3,2*d/3,1)]]

#d=180
#Ph=sf.Phantom_Str(d=d)
#Eval_Regions=[[slice(148,167,1),slice(10,130,1)],[slice(16,35,1),slice(10,130,1)]]
#PhER=sf.DrawEvalRegions(Ph,Eval_Regions)
#sf.Disp(PhER)

#%% Define MC parameters

n_mc_iter=100
n_iter=3
V_Sr=[2,3,5]
Filters=['CH','CK','CS','DH','DK','DS']
V_eta=[0.9,0.95,0.99]

#n_mc_iter=5
#n_iter=1
#V_Sr=[2]
#Filters=['CH','DH']
#Filters=['CH']
#V_eta=[0.99]

n_Filters=len(Filters)
n_R=len(Eval_Regions)
n_eta=len(V_eta)
n_Sr=len(V_Sr)

#%% initialize
resultsStr=zeros([n_mc_iter,n_Filters,n_eta,n_Sr,n_iter,2*n_R+1])

localtime = time.localtime(time.time())
print "Local current time :", localtime
i=0

#%% run

def F(Filter,i):
    resultsStr[:,i:(i+1),:,:,:,:]=sf.MC_FiltersEval(Ph,n_mc_iter=n_mc_iter,looks=3,n_iter=n_iter,V_Sr=V_Sr,V_eta=V_eta,Filters=Filter,Eval_Regions=Eval_Regions,base_fname='MC_Str500')

# Create new threads as follows
try:
   for i in range(len(Filters)):
       H=[Filters[i]]
       t = threading.Thread(target = F, kwargs = {'Filter':H,'i':i})
       t.daemon = True
       t.start()
except:
   print "Error: unable to start thread"


#%% save

resultsStr.tofile('resultsCirc100_.out')
print "Start time :", localtime
localtime = time.localtime(time.time())
print "Finish time :", localtime


#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%#%%

import thread
import time

# Define a function for the thread
def print_time( threadName, delay):
   count = 0
   while count < 5:
      time.sleep(delay)
      count += 1
      print "%s: %s" % ( threadName, time.ctime(time.time()) )

# Create two threads as follows
try:
   thread.start_new_thread( print_time, ("Thread-1", 2, ) )
   thread.start_new_thread( print_time, ("Thread-2", 4, ) )
except:
   print "Error: unable to start thread"

while 1:
   pass
#%%


n_Filters=len(Filters)
n_R=len(Eval_Regions)
n_eta=len(V_eta)
n_Sr=len(V_Sr)
#results=zeros([n_mc_iter,n_Filters,n_eta,n_Sr,n_iter,2*n_R+1])
n=100*3*3*3*6*5
f=open("/home/rgrimson/Projects/resultsCirc100.out",'rb')
results=array(struct.unpack('d'*n, f.read(8*n)))
results=results.reshape([n_mc_iter,n_Filters,n_eta,n_Sr,n_iter,2*n_R+1])
R=results[:,:,0,0,0,0]
R.mean(axis=0)