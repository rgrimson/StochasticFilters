# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:41:51 2015

@author: rgrimson
"""
import stochastic_filters as sf

src="/home/rgrimson/Projects/Lagunita/Lagunita.tif"
dst="/home/rgrimson/Projects/Lagunita/Lagunita_Filt.tif"
#load image
L=sf.loadBand(src,band=0)
#convert it to 64 bit
Img=array(L,dtype=float64)
#reserve space for the filtered result
Img_Filt=Img.copy()
#Filter Img
#extern "C" void Filter(double I_Corr[],double I_Filt[],int dimx, int dimy, int Er, int Sr, double eta, char FiltType)
sf.Filter(Img,Img_Filt,Img.shape[1],Img.shape[0],1, 5, 0.99, 'H')
#Show Images
figure()
sf.Disp(Img)
figure()
sf.Disp(Img_Filt)
#save Filtered image using original georeference
sf.Save_GTIFF(src,dst,Img_Filt)
