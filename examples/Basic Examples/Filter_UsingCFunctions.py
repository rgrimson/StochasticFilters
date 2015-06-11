# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 19:45:10 2015

@author: rgrimson
"""
import stochastic_filters as sf
import numpy as np


Img=sf.Speckle_Img(sf.Phantom_circ(d=150),looks=3)
I_Filt=np.zeros(Img.shape,dtype='float64')
sf.Filter(Img, I_Filt,Img.shape[1],Img.shape[0],1,2,0.97,'H')


#IR=sf.Speckle_Img(sf.Phantom_circ(d=500),looks=3)
#IF=sf.Filt(IR,'CH')
