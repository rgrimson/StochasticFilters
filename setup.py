from distutils.core import setup, Extension

module1 = Extension('stochastic_filters/libstochasticfilters', sources = ['src/libstochasticfilters.cpp'],include_dirs = ['src/'], 
	extra_compile_args=['-Ofast','-flto','-march=native','-funroll-loops','-std=c++11','-fPIC'],libraries=['gsl','gslcblas','m'] )
 
setup (name = 'stochastic_filters',
        version = '1.0',
        description = 'Stochastic Filter Package',
        ext_modules = [module1],
        packages = ['stochastic_filters'])
