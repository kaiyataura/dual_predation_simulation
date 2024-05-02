from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("QuadTree.pyx", annotate=True)
)