from stochastic_filter_functions import *
#from c_functions import *

import ctypes
from numpy.ctypeslib import ndpointer

float_t = ctypes.c_float
double_t = ctypes.c_double
int_t = ctypes.c_int
char_t = ctypes.c_char

#test_func = ctypes.cdll.LoadLibrary(__path__[0]+"/libstochasticfilters.so").test_func
#test_func.restype = float_t
#test_func.argtypes = [float_t]

# digamma = ctypes.cdll.LoadLibrary(__path__[0]+"/libstochasticfilters.so").digamma_
# digamma.restype = double_t
# digamma.argtypes = [double_t]

# polygamma = ctypes.cdll.LoadLibrary(__path__[0]+"/libstochasticfilters.so").polygamma_
# polygamma.restype = double_t
# polygamma.argtypes = [int_t,double_t]

NR_L = ctypes.cdll.LoadLibrary(__path__[0]+"/libstochasticfilters.so").NR_L
NR_L.restype = double_t
NR_L.argtypes = [double_t]



Compute_ML_Param = ctypes.cdll.LoadLibrary(__path__[0]+"/libstochasticfilters.so").Compute_ML_Param
Compute_ML_Param.restype = None
Compute_ML_Param.argtypes = [ndpointer(double_t,flags='C_CONTIGUOUS'),
                              ndpointer(double_t,flags='C_CONTIGUOUS'),
                              ndpointer(double_t,flags='C_CONTIGUOUS'),
                              ndpointer(double_t,flags='C_CONTIGUOUS'),
                              int_t,
                              int_t,
                              int_t]


#Computes the Maximal Liklihood parameters (S for mean, L for n.looks) for each pixel in a given image M, using descented windows, and the dimensions of the image. returns also the number of pixels in the most likely window.
Compute_ML_Param_NMWin = ctypes.cdll.LoadLibrary(__path__[0]+"/libstochasticfilters.so").Compute_ML_Param_NMWin
Compute_ML_Param_NMWin.restype = None
Compute_ML_Param_NMWin.argtypes = [ndpointer(double_t,flags='C_CONTIGUOUS'),
                            ndpointer(double_t,flags='C_CONTIGUOUS'),
                          ndpointer(double_t,flags='C_CONTIGUOUS'),
                          ndpointer(double_t,flags='C_CONTIGUOUS'),
                          int_t,
                          int_t]

Filter_fromMLParam = ctypes.cdll.LoadLibrary(__path__[0]+"/libstochasticfilters.so").Filter_fromMLParam
Filter_fromMLParam.restype = None
Filter_fromMLParam.argtypes = [ndpointer(double_t,flags='C_CONTIGUOUS'),
                            ndpointer(double_t,flags='C_CONTIGUOUS'),
                            ndpointer(double_t,flags='C_CONTIGUOUS'),
                            ndpointer(double_t,flags='C_CONTIGUOUS'),
                            ndpointer(double_t,flags='C_CONTIGUOUS'),
                            int_t,
                            int_t,
                            int_t,
                            double_t,
                            char_t]

Filter = ctypes.cdll.LoadLibrary(__path__[0]+"/libstochasticfilters.so").Filter
Filter.restype = None
Filter.argtypes = [ndpointer(double_t,flags='C_CONTIGUOUS'),
                   ndpointer(double_t,flags='C_CONTIGUOUS'),
                   int_t,
                   int_t,
                   int_t,
                   int_t,
                   double_t,
                   char_t]

Filter_and_Param = ctypes.cdll.LoadLibrary(__path__[0]+"/libstochasticfilters.so").Filter
Filter_and_Param.restype = None
Filter_and_Param.argtypes = [ndpointer(double_t,flags='C_CONTIGUOUS'),
		                     ndpointer(double_t,flags='C_CONTIGUOUS'),
		                     ndpointer(double_t,flags='C_CONTIGUOUS'),
		                     ndpointer(double_t,flags='C_CONTIGUOUS'),
		                     ndpointer(double_t,flags='C_CONTIGUOUS'),
		                     int_t,
		                     int_t,
		                     int_t,
		                     int_t,
		                     double_t,
		                     char_t]




