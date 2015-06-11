# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:57:16 2015

@author: rgrimson
"""

print tabulate(table, headers=["Planet","R (km)", "mass (x 10^29 kg)"],tablefmt="latex",floatfmt=".4f")

#%%   
   
Phant='O'
eta=0.97
Sr=2
n_mc_it=10
Ph=Phantom_circ(40)
Eval_Regions=[[slice(0,7,1),slice(0,40,1)],[slice(12,28,1),slice(12,28,1)]]
#%%

R=MC_FiltersEval(Ph,n_mc_iter=n_mc_it,n_iter=2,Filters=['CK','DK','CH','DH','CS','DS'],Eval_Regions=Eval_Regions,base_fname='QO')
#%%

RC10=MC_FiltersEval(Ph,n_mc_iter=n_mc_it,n_iter=2,Filters=['CK','CH','CS'],Eval_Regions=Eval_Regions,base_fname='QO')
#%%
RD10=MC_FiltersEval(Ph,n_mc_iter=n_mc_it,n_iter=2,Filters=['DK','DH','DS'],Eval_Regions=Eval_Regions,base_fname='QO')
#%%

n_Filters=6
n_R=len(Eval_Regions)
n_eta=1
n_Sr=1
n_iter=2
D=zeros([n_mc_it,n_Filters,n_eta,n_Sr,n_iter,2*n_R+1])
D[:,0:3]=RC100
D[:,3:6]=RD100
F=D.reshape([n_mc_it,60])
E=D
D=F

Filt="Sh_"
Phant='O'

fname="QO_"

#%%
fig = plt.figure(1, figsize=(90, 60))


# Create an axes instance
ax = fig.add_subplot(111)

bp=ax.boxplot(D[:,arange(0,56,5)+1], notch=0, sym='.', vert=1, whis=1.5,  patch_artist=True)
#bp=ax.boxplot(D[:,linspace[0,60,3]], notch=0, sym='.', vert=1, whis=1.5,  patch_artist=True)
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

fig.show()
#%%


# Save the figure
fig.savefig(fname+'.png', bbox_inches='tight')
