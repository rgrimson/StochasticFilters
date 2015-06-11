# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 20:08:33 2015

@author: rgrimson
"""
#import stochastic_filters as sf
import struct
import numpy as np

n_mc_iter=100
Filters=['CH','CK','CS','DH','DK','DS']
#Filters=['CH','DH']
V_eta=[0.9,0.95,0.99]
V_Sr=[2,3,5]
n_iter=3

d=180
Eval_Regions=[[slice(0,d/7,1),slice(0,d,1)],[slice(d/3,2*d/3,1),slice(d/3,2*d/3,1)]]
r=100
d=5*r
m=3
Ph=sf.Phantom_4SQ(d=d)
#Eval for 4SQ
Eval_Regions=[[slice(r+m+1,2*r-m+1,1),slice(r+m+1,2*r-m+1,1)],[slice(r+m+1,2*r-m+1,1),slice(3*r+m+1,4*r-m+1,1)],[slice(3*r+m+1,4*r-m+1,1),slice(r+m+1,2*r-m+1,1)],[slice(3*r+m+1,4*r-m+1,1),slice(3*r+m+1,4*r-m+1,1)],[slice(0,d/7,1),slice(0,d/2-m,1)]]



n_Filters=len(Filters)
n_R=len(Eval_Regions)
n_eta=len(V_eta)
n_Sr=len(V_Sr)
#results=zeros([n_mc_iter,n_Filters,n_eta,n_Sr,n_iter,2*n_R+1])
n=n_mc_iter*n_Filters*(2*n_R+1)*n_eta*n_Sr*n_iter
f=open("/home/rgrimson/graph/results4SQ.out",'rb')
results=np.array(struct.unpack('d'*n, f.read(8*n)))
R=results.reshape([n_mc_iter,n_Filters,n_eta,n_Sr,n_iter,2*n_R+1])


#%%
def myplot():
    global R, Plot_iter,Plot_eta, Plot_Sr, Plot_Filters, Filters, V_eta, V_Sr,q,fname,path
    iPlot_iter=np.array(Plot_iter)-1
    iPlot_Filters=[]
    for F in Plot_Filters:
        iPlot_Filters.append(Filters.index(F))
    iPlot_eta=[]
    for e in Plot_eta:
        iPlot_eta.append(V_eta.index(e))
    iPlot_Sr=[]
    for r in Plot_Sr:
        iPlot_Sr.append(V_Sr.index(r))
        
    n_pFilters=len(Plot_Filters)
    n_pSr=len(Plot_Sr)
    n_peta=len(Plot_eta)
    n_piter=len(Plot_iter)
    
    P=np.zeros([n_mc_iter,n_pFilters,n_peta,n_pSr,n_piter])
    nF=0
    for iF in iPlot_Filters:
      ne=0
      for ie in iPlot_eta:
        nr=0  
        for ir in iPlot_Sr:
          ni=0  
          for ii in iPlot_iter:  
              P[:,nF,ne,nr,ni]=R[:,iF,ie,ir,ii,q]
              ni+=1
          nr+=1
        ne+=1
      nF=nF+1
      
    P=P.reshape([n_mc_iter,n_pFilters*n_peta*n_pSr*n_piter])
    
        
    # Create a figure instance
    D=P#R[:,iPlot_Filters,iPlot_eta,iPlot_Sr,iPlot_iter,0].reshape([n_mc_iter,n_pFilters*n_pSr*n_peta*n_pSr])
    lab=[]
    for Filt in Plot_Filters:
        for eta in Plot_eta:
            for Sr in Plot_Sr:
                for i in Plot_iter:
                    lab.append(Filt+'_'+'e'+str(int(eta*100))+'_'+'r'+str(int(2*Sr+1))+'_'+'it'+str(int(i)))
                    
    fig = plt.figure(1)
    fig.set_size_inches(18.5,6.0)
    
    # Create an axes instance
    ax = fig.add_subplot(111)
    ax.cla()
    
    bp=ax.boxplot(D, notch=0, sym='.', vert=1, whis=1.5,  patch_artist=True)
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#204e6f', linewidth=2)
        # change fill color
        box.set( facecolor = '#fe9ba0' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#dc7e99', linewidth=1)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#1f243a', linewidth=1)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#4a94b9', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#dc7e99', alpha=1)
    
    ## Custom x-axis labels
    ax.set_xticklabels(lab,rotation=90)
    
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    fig.savefig(path+fname, bbox_inches='tight',dpi=80)
    print "Fin!"

#%%


#%%
#Plot_Filters=['CH','CK','CS','DH','DK','DS'] #desired plot filters
#Plot_eta=[0.9] #desired plot etas
#Plot_Sr=[2,5] #desired plot radius
#Plot_iter=[1,3] #desired plot iterations

path="/home/rgrimson/graph/4SQ/"
fname='Q_Centered.png'
q=0 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['CH','CK','CS'] #desired plot filters
Plot_eta=[0.9,0.95,0.99] #desired plot etas
Plot_Sr=[2,3,5] #desired plot radius
Plot_iter=[1,2,3] #desired plot iterations

myplot()

fname='Q_Descentered.png'
q=0 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['DH','DK','DS'] #desired plot filters
Plot_eta=[0.9,0.95,0.99] #desired plot etas
Plot_Sr=[2,3,5] #desired plot radius
Plot_iter=[1,2,3] #desired plot iterations

myplot()

fname='Q_All.png'
q=0 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['CH','CK','CS','DH','DK','DS'] #desired plot filters
Plot_eta=[0.9] #desired plot etas
Plot_Sr=[5] #desired plot radius
Plot_iter=[3] #desired plot iterations

myplot()

fname='P_Centered.png'
q=1 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['CH','CK','CS'] #desired plot filters
Plot_eta=[0.9,0.95,0.99] #desired plot etas
Plot_Sr=[2,3,5] #desired plot radius
Plot_iter=[1,2,3] #desired plot iterations

myplot()

fname='P_Descentered.png'
q=1 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['DH','DK','DS'] #desired plot filters
Plot_eta=[0.9,0.95,0.99] #desired plot etas
Plot_Sr=[2,3,5] #desired plot radius
Plot_iter=[1,2,3] #desired plot iterations

myplot()

fname='P_All.png'
q=1 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['CH','CK','CS','DH','DK','DS'] #desired plot filters
Plot_eta=[0.9] #desired plot etas
Plot_Sr=[5] #desired plot radius
Plot_iter=[3] #desired plot iterations

myplot()

fname='L_Centered.png'
q=2 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['CH','CK','CS'] #desired plot filters
Plot_eta=[0.9,0.95,0.99] #desired plot etas
Plot_Sr=[2,3,5] #desired plot radius
Plot_iter=[1,2,3] #desired plot iterations

myplot()

fname='L_Descentered.png'
q=2 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['DH','DK','DS'] #desired plot filters
Plot_eta=[0.9,0.95,0.99] #desired plot etas
Plot_Sr=[2,3,5] #desired plot radius
Plot_iter=[1,2,3] #desired plot iterations

myplot()

fname='L_All.png'
q=2 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['CH','CK','CS','DH','DK','DS'] #desired plot filters
Plot_eta=[0.9] #desired plot etas
Plot_Sr=[5] #desired plot radius
Plot_iter=[3] #desired plot iterations

myplot()


#%%
fname='Q_H.png'
q=0 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['CH','DH'] #desired plot filters
Plot_eta=[0.9,0.95,0.99] #desired plot etas
Plot_Sr=[2,3,5] #desired plot radius
Plot_iter=[1,2,3] #desired plot iterations

myplot()

#%%
#SAVE PHANTOMS
clf()
sf.Disp(sf.Phantom_4SQ(d=500))
path="/home/rgrimson/graph/4SQ/"
fname="Ph4SQ"
savefig(path+fname, bbox_inches='tight')

clf()
sf.Disp(sf.Phantom_circ(d=180))
path="/home/rgrimson/graph/Circ/"
fname="PhCirc"
savefig(path+fname, bbox_inches='tight')

clf()
sf.Disp(sf.Phantom_Str(d=180))
path="/home/rgrimson/graph/Str/"
fname="PhStr"
savefig(path+fname, bbox_inches='tight')

#%%
R=results.reshape([100,1,1,1,1,11])
n_mc_iter=100
Filters=['NLS']
#Filters=['CH','DH']
V_eta=[2]
V_Sr=[5]
n_iter=1

path="/home/rgrimson/Dropbox/Compartidas/Frery_SDNLM/MC/"
fname='L_nlsar.png'
q=2 #0 for Q-index, 1 for RadPres Reg1, 2 for nLooks Reg1, 3 for RadPres Reg2, 4 for nLook Reg2, etc 

Plot_Filters=['NLS'] #desired plot filters
Plot_eta=[2] #desired plot etas
Plot_Sr=[5] #desired plot radius
Plot_iter=[1] #desired plot iterations

myplot()
