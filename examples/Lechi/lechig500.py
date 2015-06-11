# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:41:51 2015

@author: rgrimson
"""
#%%
import gdal
import stochastic_filters as sf

def Save_GTIFF(src_filename, dst_filename,data):
    format = "GTIFF"
    driver = gdal.GetDriverByName( format )
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    dst_ds = driver.CreateCopy( dst_filename, dataset, 0 )
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds  = None

def Save_Stk_GTIFF(src_filename, dst_filename,data):
    format = "GTIFF"
    driver = gdal.GetDriverByName( format )
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    dst_ds = driver.CreateCopy( dst_filename, dataset, 0 )
    for i in range(data.shape[0]):
        dst_ds.GetRasterBand(i+1).WriteArray(data[i])
    dst_ds  = None

def loadStack(src_filename):
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    tifArray = dataset.ReadAsArray()
    return array(tifArray,dtype=float64)

def FiltStack(Stk, Filt='CK', Er=1, Sr=2, eta=0.95):
    d=Stk.shape[0]
    F=Stk.copy()
    for i in range(d):
        print ("Filtering Band:",i)
        F[i,:,:]=sf.Filt(Stk[i,:,:],Filt, Er=Er, Sr=Sr, eta=eta)
    return F

#%%  
src='/home/rgrimson/Projects/Lechig/Images/CSK_HImage_stack_lechig_sigma0'
Stk=loadStack(src)

#for Filt in ['CK','CH','CS','DK','DH','DS',]:
#    F=FiltStack(Stk,Filt=Filt)
#    Save_Stk_GTIFF(src, src+'_'+Filt.tif,F)
#
#src='/home/rgrimson/Projects/lechi500/CSK_HImage_stack_lechig500_sigma0'
#Stk=loadStack(src)
#
#for Filt in ['CK','CH','CS','DK','DH','DS',]:
#    F=FiltStack(Stk,Filt=Filt)
#    Save_Stk_GTIFF(src, src+'_'+Filt+'.tif',F)
#    
#### PYTHON 1
#
#for Filt in ['CK','CH','CS']:
#    F=FiltStack(Stk,Filt=Filt)
#    Save_Stk_GTIFF(src, src+'_'+Filt+'.tif',F)
#    
#### PYTHON 2

for Filt in ['CH']:
    F=FiltStack(Stk,Filt=Filt)
    Save_Stk_GTIFF(src, src+'_'+Filt+'_Pow.tif',F)
    F_dB=10*log10(F)
    Save_Stk_GTIFF(src, src+'_'+Filt+'_dB.tif',F_dB)
    
#%%  

#src='/home/rgrimson/Projects/Lechig/Images/Lechig_dB'
#src='/home/rgrimson/Projects/Lechig/Images/CSK_HImage_stack_lechig_sigma0'
src='/home/rgrimson/Projects/Lechig/Lechi500.tif'
dst='/home/rgrimson/Projects/Lechig/Lechi500_dB.tif'

Stk=loadStack(src)
Stk=10*log10(Stk)
Save_Stk_GTIFF(src,dst,Stk)

src='/home/rgrimson/Projects/Lechig/Images/Lechig_CK_Pow.tif'
dst='/home/rgrimson/Projects/Lechig/Images/Lechig_CK_dB.tif'

Stk=loadStack(src)
Stk=10*log10(Stk)
Save_Stk_GTIFF(src,dst,Stk)

src='/home/rgrimson/Projects/Lechig/Images/Lechig_CS_Pow.tif'
dst='/home/rgrimson/Projects/Lechig/Images/Lechig_CS_dB.tif'

Stk=loadStack(src)
Stk=10*log10(Stk)
Save_Stk_GTIFF(src,dst,Stk)



#%%
#
#
#
#
#
#
#
#sf.Disp(Sigma0_b7)
#sf.Disp(10*log(Sigma0_b1))
#
#Filt_b1_CK=sf.Filt(Sigma0_b1,'CK', Er=1, Sr=2, eta=0.95)
#figure() #abre nueva figura
#sf.Disp(10*log(Filt_b1_CK))
#
##### estos son los comandos en C
#B_Filt=B.copy()
##extern "C" void Filter(double I_Corr[],double I_Filt[],int dimx, int dimy, int Er, int Sr, double eta, char FiltType)
#sf.Filter(B,B_Filt,B.shape[1],B.shape[0],0, 2, 0.97, 'S')
#
#figure()
#sf.Disp(B)
#figure()
#sf.Disp(B_Filt)