import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy

# cythonize the interface code
extensions = [
    Extension(
        "coaddit.r3d._r3d_interface",
        ["coaddit/r3d/_r3d_interface.pyx",
         "coaddit/r3d/r2d.c",
         "coaddit/r3d/v2d.c"],
        include_dirs=[numpy.get_include()]
    ),
]

with open(os.path.join(os.path.dirname(__file__), "README.md")) as fp:
    long_description = fp.read()

setup(
    name="coaddit",
    version="0.1.0",
    packages=find_packages(),
    setup_requires=['numpy', 'cython'],
    install_requires=['numpy'],
    include_package_data=True,
    author="Matthew R. Becker",
    author_email="becker.mr@gmail.com",
    description=(
        "coadd two-dimensional images using Devon Powell's r3d exact "
        "voxelization code (https://github.com/devonmpowell/r3d)"),
    license="BSD-3-Clause",
    url="https://github.com/beckermr/coaddit",
    long_description=long_description,
    long_description_content_type='text/markdown; charset=UTF-8; variant=GFM',
    zip_safe=False,
    ext_modules=cythonize(extensions),
)
