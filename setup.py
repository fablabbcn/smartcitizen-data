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

setup(
    name='scdata',
    version= '0.6.4',
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

    install_requires=[
                        'branca==0.4.0',
                        'Flask~=1.1.2',
                        'folium==0.12.1',
                        'geopy==1.21.0',
                        'Jinja2==2.11.2',
                        'matplotlib==3.2.1',
                        'numpy~=1.20.0',
                        'pandas>=1.3.0',
                        'pdfrw==0.4',
#                       'PDPbox==0.2.0',
                        'plotly~=4.14.3',
                        'PyYAML==5.3.1',
                        'requests==2.23.0',
                        'reportlab==3.5.*',
                        'scipy~=1.5.0',
                        'scikit-learn~=1.0.1',
                        'seaborn==0.10.1',
                        'termcolor==1.1.0',
                        'tqdm~=4.50.2',
                        'tzwhere==3.0.3',
                        'urllib3==1.25.9'
                    ],

    setup_requires=['wheel'],

    include_package_data=True,
    zip_safe=False
)
