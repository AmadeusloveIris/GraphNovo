import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [Extension("edge_matrix_gen", ["edge_matrix_gen.pyx"])]

setup(
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}, include_path = [numpy.get_include()])
)
