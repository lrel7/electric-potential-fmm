from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension("fmm_par", ["fmm_par.pyx"], extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"])
]

setup(
    ext_modules = cythonize(extensions)
)
