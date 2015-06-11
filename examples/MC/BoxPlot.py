# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 11:04:56 2015

@author: rgrimson
"""
import os
os.chdir('/home/rgrimson/Dropbox/Compartidas/Frery_SDNLM/Code/Filters')
from StochasticFilters import *
os.chdir('/home/rgrimson/Projects/SDNLM/CompareClassif')

#%%
Phantom=Phantom_circ(40)
Phant="circ"
Filt="S"
n_mc_it=100
n_f_iter=4
Sr=5
eta=0.97
kDD=zeros([n_mc_it,n_f_iter+1])
kCD=zeros([n_mc_it,n_f_iter+1])
kDC=zeros([n_mc_it,n_f_iter+1])
kCC=zeros([n_mc_it,n_f_iter+1])
F="C"+Filt+'_'
VLS_CS,kCC,kDC=PhantomCompareFiltersClassifiers(Phantom, n_mc_it=n_mc_it, n_f_it=n_f_iter, Sr=Sr, eta=eta, Filt=F, Eval_Regions=[])
fname="kappa_" + F + Phant + str(int(eta*100))+"X" + str(2*Sr+1) +" _it" + str(n_mc_it)
savetxt(fname+'_D.txt',kDC,delimiter=',')
savetxt(fname+'_C.txt',kCC,delimiter=',')
F="D"+Filt+'_'
VLS_DS,kCD,kDD=PhantomCompareFiltersClassifiers(Phantom, n_mc_it=n_mc_it, n_f_it=n_f_iter, Sr=Sr, eta=eta, Filt=F, Eval_Regions=[])
fname="kappa_" + F + Phant + str(int(eta*100))+"X" + str(2*Sr+1) +" _it" + str(n_mc_it)
savetxt(fname+'_D.txt',kDD,delimiter=',')
savetxt(fname+'_C.txt',kCD,delimiter=',')

D=zeros([n_mc_it,4*(n_f_iter+1)])

for i in range(n_f_iter+1):
    D[:,i] =kCC[:,i]
    D[:,n_f_iter+i] =kCD[:,i]
    D[:,2*n_f_iter+i] =kDC[:,i]
    D[:,3*n_f_iter+i] =kDD[:,i]

#%%

kCCK_S2=loadtxt("kCCK_S5.txt",delimiter=',')
kCDK_S2=loadtxt("kCDK_S5.txt",delimiter=',')
kDCK_S2=loadtxt("kDCK_S5.txt",delimiter=',')
kDDK_S2=loadtxt("kDDK_S5.txt",delimiter=',')
   
D=zeros([100,12])
D[:,0] =kCCK_S2[:,0]
D[:,1] =kCCK_S2[:,1]
D[:,2] =kCCK_S2[:,2]
D[:,3] =kCDK_S2[:,0]
D[:,4] =kCDK_S2[:,1]
D[:,5] =kCDK_S2[:,2]
D[:,6] =kDCK_S2[:,0]
D[:,7] =kDCK_S2[:,1]
D[:,8] =kDCK_S2[:,2]
D[:,9] =kDDK_S2[:,0]
D[:,10]=kDDK_S2[:,1]
D[:,11]=kDDK_S2[:,2]
#%%
D=D[0:90,0:12]
# Create a figure instance
Filt="Sh_"
Phant='O'

fname="kappa_" + Filt + Phant + str(int(eta*100))+"X" + str(2*Sr+1) +" _it" + str(n_mc_it)

fig = plt.figure(1, figsize=(90, 60))

# Create an axes instance
ax = fig.add_subplot(111)

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
ax.set_xticklabels(['C_'+Filt+'C0it','C_'+Filt+'C1it','C_'+Filt+'C2it','C_'+Filt+'D0it','C_'+Filt+'D_1it','C_'+Filt+'D2it','D_'+Filt+'C0it','D_'+Filt+'C1it','D_'+Filt+'C2it','D_'+Filt+'D0it','D_'+Filt+'D1it','D_'+Filt+'D2it'])

## Remove top axes and right axes ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Save the figure
fig.savefig(fname+'.png', bbox_inches='tight')
fig.show()
#%%

#Grand Budapest Hotel Colors
#4a94b9 #celeste moño
#8b9b9a #azul celeste
#204e6f #azul lindo
#1f243a #azul oscuro
#372728 #negro rojizo
#8e4d31 #marron oscuro
#a8715c #marron claro
#f0c09c #cremita marron
#ecbca5 #cremita marron-grisaceo
#fe9ba0 #rosa claro
#dc7e99 #rosa oscuro
#9d767b #violeta

#from matplotlib.colors import from_levels_and_colors
#from matplotlib.colors import hex2color
#OLD BPH Colors
#hex2color("#060604"),
#hex2color("#610c07"),
#hex2color("#a96d51"),
#hex2color("#e69c81"),#FACE
#hex2color("#f9bec2"),
#hex2color("#d1c4bc"),
#hex2color("#4a94b9"),#Whiskers
#hex2color("#9f3645"),#BOX BORDER
#hex2color("#e69c81")



gradient = np.zeros([256,3])
base_colors=array([
hex2color("#372728"), #negro rojizo
hex2color("#8e4d31"), #marron oscuro
hex2color("#a8715c"), #marron claro
hex2color("#f0c09c"), #cremita marron
hex2color("#ecbca5"), #cremita marron-grisaceo
hex2color("#fe9ba0"), #rosa claro
hex2color("#dc7e99"), #rosa oscuro
hex2color("#9d767b"),#violeta
hex2color("#8b9b9a"), #azul celeste grisaceo
hex2color("#4a94b9"), #celeste moño
hex2color("#204e6f"), #azul lindo
hex2color("#1f243a")]) #azul oscuro

for j in range(11):
    for i in range(23):    
        gradient[23*j+i,:]=((23-i)*base_colors[j]+i*base_colors[j+1])/23.0
cmap, norm = from_levels_and_colors(linspace(0,255,255),gradient[0:254,:],extend='neither')    
imshow(A,cmap=cmap,vmin=0,vmax=253)