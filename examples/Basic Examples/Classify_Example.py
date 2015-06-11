#!/usr/bin/python2.7
import stochastic_filters as sf
import numpy as np
Ph=sf.Phantom_circ(d=100)
PhC, K=sf.ClassifyPhantom(Ph)
I_Corr=sf.Speckle_Img(Ph,looks=4)
I_Class=sf.Classify(I_Corr,K=K,mode='C')
I_Filt=sf.Filt(I_Corr,'CK')


sf.Disp(PhC)
sf.Disp(I_Class)

print 'kappa: ',sf.kappa2(PhC,I_Class)
print 'Q: ',sf.Q(Ph,I_Filt)


#sf.FiltersEval(I_Corr,I_Orig=Ph,n_iter=1,V_Sr=[2],V_eta=[0.97],Filters=["CK"],Eval_Regions=[])


