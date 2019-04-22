import os
from setuptools import setup, find_packages


with open(os.path.join(os.path.dirname(__file__), "README.md")) as fp:
    long_description = fp.read()

setup(
    name="coaddit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=['numpy', 'cython'],
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
)
