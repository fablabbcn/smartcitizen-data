# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

import sys

if sys.version_info < (3,0):
    sys.exit("scdata requires python 3.")

PROJECT_URLS = {
    "Documentation": "https://docs.smartcitizen.me/",
    "Source Code": "https://github.com/fablabbcn/smartcitizen-data",
}

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name='scdata',
    version='1.0.1',
    description='Analysis of sensors and time series data',
    author='oscgonfer',
    license='GNU-GPL3.0',
    packages=find_packages(),
    keywords=['air', 'sensors', 'Smart Citizen'],
    url='https://github.com/fablabbcn/smartcitizen-data',
    project_urls=PROJECT_URLS,
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[REQUIREMENTS],
    setup_requires=['wheel'],
    python_requires=">=3.9",
    include_package_data=True,
    zip_safe=False
)
