# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

from scdata import __version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='scdata',
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version= __version__,
    description='Smart Citizen Data',
    author='oscgonfer',
    license='GNU-GPL3.0',
    packages=find_packages(),

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[  'attrs==19.3.0',
                        'branca==0.4.0',
                        'certifi==2020.4.5.1',
                        'chardet==3.0.4',
                        'cycler==0.10.0',
                        'Flask==1.1.2',
                        'folium==0.11.0',
                        'geographiclib==1.50',
                        'geopy==1.21.0',
                        'idna==2.9',
                        'importlib-metadata==1.6.0',
                        'Jinja2==2.11.2',
                        'kiwisolver==1.2.0',
                        'MarkupSafe==1.1.1',
                        'matplotlib==3.2.1',
                        'numpy==1.18.3',
                        'pandas==1.0.3',
                        'PDPbox==0.2.0',
                        'plotly==4.6.0',
                        'pyparsing==2.4.7',
                        'python-dateutil==2.8.1',
                        'pytz==2020.1',
                        'PyYAML==5.3.1',
                        'requests==2.23.0',
                        'scipy==1.4.1',
                        'scikit-learn==0.22.2.post1',
                        'seaborn==0.10.1',
                        'Shapely==1.7.0',
                        'six==1.14.0',
                        'termcolor==1.1.0',
                        'tzwhere==3.0.3',
                        'urllib3==1.25.9',
                        'webencodings==0.5.1',
                        'zipp==3.1.0'],
)
