import glob
import shutil
from setuptools import setup, find_packages

# define and read requirements
REQUIRES_PATH = 'requirements.txt'

with open(REQUIRES_PATH, 'r') as file:
    INSTALL_REQUIRES = file.read().splitlines()

PYTHON_REQUIRES = ">= 3.10, < 4" 

# get project version
with open('VERSION.txt', 'r') as file:
    VERSION = file.read()

# define license file
LICENSE = 'LICENSE.txt'

# remove __pycache__ directories
pycache_dirs = glob.glob('**/__pycache__', recursive=True)

for pycache_dir in pycache_dirs:
    shutil.rmtree(pycache_dir)


setup(
    # general properties
    name='OpenFOAMPost', 
    version=VERSION, 
    description='A powerful tool to to post-process OpenFOAM simulations.', 
    url='https://github.com/TheBusyDev/OpenFOAMPost',
    author='TheBusyDev', 
    author_email='pietro.busy@gmail.com',
    license=LICENSE,

    # python version
    python_requires=PYTHON_REQUIRES,

    # include packages
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=INSTALL_REQUIRES,

    # add entry point to run script in command line
    entry_points={'console_scripts': 'ofpost=ofpost:__main__'}
)