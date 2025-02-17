# OpenFOAMPost
A powerful tool to post-process **OpenFOAM** simulations (written in *Python* 🐍). \
Are you fed up with processing OpenFOAM simulations using *ParaView*? \
Yeah, OpenFOAM *function objects* are awesome🌟, however processing .vtk manually can be very boring👎.

For this purpose, **OpenFOAMPost** can extract *colorful images* 🌈 from your simulations!!


## How to install
The following instructions are well-tested on Linux-based systems 🐧.

### Dependencies
Before starting, install all the dependencies:
- **Ubuntu**
```
sudo apt install wget python3-pip pipx
```
- **Fedora**
```
sudo dnf install wget python3-pip pipx
```

### Install package
You can install the wheel package through *pipx* (this is the suggested option) 🚀:
```
wget https://github.com/TheBusyDev/OpenFOAMPost/releases/download/v1.0.0/ofpost-1.0.0-py3-none-any.whl
pipx install ofpost-1.0.0-py3-none-any.whl
pipx ensurepath
```
or, alternatively, through *pip* (not supported starting from Ubuntu 24.04):
```
wget https://github.com/TheBusyDev/OpenFOAMPost/releases/download/v1.0.0/ofpost-1.0.0-py3-none-any.whl
pip install ofpost-1.0.0-py3-none-any.whl
```


## How to use
This script will essentially look for .vtk, .dat, .xy files into the current directory and convert them into .png format (other formats can be selected by the user).

⚠️**IMPORTANT**: this script will look into the current directory and its *subdirectories* **recursively**!! Be sure you are in the correct directory before launching it!

**BASIC USAGE**: 
```
ofpost
```

Other options can be specified by the user. All the input arguments can be listed as:
```
ofpost --help
```

For instance, the user can post-process a 2D, steady-state, incompressible simulation by typing:
```
ofpost --case 2D --steady yes --incomp yes
```


## How to contribute
All sorts of contributions are well-welcomed 🤗! You can start by cloning the git repo:
```
git clone https://github.com/TheBusyDev/OpenFOAMPost.git
```
Then all the necessary variables can be initialized by executing:
```
cd OpenFOAMPost
source init.sh
```

⚠️**NOTE**: `source init.sh` command is always required before starting to work on this project in order to initialize all the enviromnent variables properly!

Hence, by calling :
```
ofpost
```
the program will be launched into the [/test](/test) folder or its subdirectories!

Moreover, the following command can be used to clean up the [/test](/test) folder:
```
ofclean
```

All the source files can be found in [/src/ofpost](/src/ofpost) directory! Enjoy! 🤓


## Examples
Many other examples can be found inside [/test](/test) folder! 🌈

![slice](/test/postProcessing/VelocitySlice/zNormalPlane_U_mag_0.5.png)
![residuals](/test/postProcessing/Residuals/residuals.png)
