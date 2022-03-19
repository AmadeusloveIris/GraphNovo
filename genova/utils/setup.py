import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize


setup(
    ext_modules = cythonize("edge_matrix_gen.pyx", compiler_directives={'language_level' : "3"}, include_path = [numpy.get_include()])
)
