"""
Build script for Cython extension modules.

Usage:
    python setup_cython.py build_ext --inplace

Or via pip:
    pip install -e ".[cython]"
"""

import numpy as np
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

if USE_CYTHON:
    extensions = cythonize(
        [
            Extension(
                "duelist_zero.env.fast_obs",
                sources=["src/duelist_zero/env/fast_obs.pyx"],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            ),
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
else:
    # Fallback: build from pre-generated .c file
    extensions = [
        Extension(
            "duelist_zero.env.fast_obs",
            sources=["src/duelist_zero/env/fast_obs.c"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]

setup(
    name="duelist-zero-cython",
    ext_modules=extensions,
    package_dir={"": "src"},
    zip_safe=False,
)
