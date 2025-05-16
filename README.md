# OpenFOAMPost - OpenFOAM Post-Processing Utility
A powerful tool to post-process **OpenFOAM** simulations (written in *Python* 🐍). \
Are you fed up with processing OpenFOAM simulations using *ParaView*? \
Yeah, OpenFOAM *function objects* are awesome🌟, however processing .vtk manually can be very boring👎.

For this purpose, **OpenFOAMPost** can extract *colorful images* 🌈 from your simulations!!

**⚠️NOTE:** If you are just a user, it is sufficient that you take a look at [install](#how-to-install) and [usage](#how-to-use) sections (you can find also an [example](#examples) section below). \
For code contributions, refer to [contribute](#how-to-contribute) and [build](#how-to-build) sections! Enjoy! 🤗


## How to install
The following instructions are well-tested on Linux-based systems 🐧. \
However, installation process is also reported for Windows and MacOS platforms. \
Dependencies installation is specified for Linux only. MacOS and Windows users must find their way to install [pipx](https://pipx.pypa.io/stable/installation/) and/or [pip](https://pip.pypa.io/en/stable/installation/).

### Dependencies - Linux only (working also on WSL)
Before starting, install all the dependencies (only procedure for *Ubuntu-based* systems is reported, find an equivalent for your specific Linux distro):
```
sudo apt update
sudo apt install python3-pip pipx
pipx ensurepath
```

**⚠️WARNING:** 
- It is necessary to restart the shell session (i.e. the terminal) to apply the changes above!
- The package requires `python3.10` or later!


### Install package - Linux, MacOS, Windows
You can install the wheel package through *pipx* (this is the suggested option) 🚀:
```
pipx install openfoampost
```
or, alternatively, through *pip* (not supported starting from Ubuntu 24.04):
```
pip install openfoampost
```


### Upgrade package
When a new version is released, the package can be upgrade, preferably through *pipx*:
```
pipx upgrade openfoampost
```
or, alternatively, through *pip*:
```
pip install --upgrade openfoampost
```


## How to use
This script will essentially look for *.vtk*, *.dat*, *.xy* files into the specified directories and convert them into *.png* format (other formats can be selected by the user).

**⚠️IMPORTANT:** This script will look into the specified directories and their *subdirectories* **recursively**!! Be sure you selected the correct directory before launching it!

**⚠️WARNING:** If you get a `command not found...` error, ensure you installed the [dependencies](#dependencies---linux-only-working-also-on-wsl) correctly!

### Basic usage 
```
ofpost /path/to/OpenFOAM/simulation
```

Other options can be specified by the user. All the input arguments can be listed by:
```
ofpost --help
```

For example, the user can post-process a 2D, steady-state, incompressible simulation in the current working directory by typing:
```
ofpost . --2D yes --steady yes --incomp yes
```


## How to contribute
All sorts of contributions are very welcome 🤗! You can start by cloning the git repo:
```
git clone https://github.com/TheBusyDev/OpenFOAMPost.git
```
Then, the necessary modules can be installed and the environment variables can be initialized by executing (a [virtual environment](https://docs.python.org/3/library/venv.html) is strongly suggested):
```
cd OpenFOAMPost
pip install -r requirements.txt
source init.sh
```

**⚠️NOTE:** `source init.sh` command is always required before starting to work on this project in order to initialize all the enviromnent variables properly!

Hence, by calling :
```
ofpost-test
```
the program will be launched into the [/test](/test) folder!

Moreover, the following command can be used to clean up the [/test](/test) folder:
```
ofpost-clean
```

All the source files can be found in [/src/ofpost](/src/ofpost) directory! Enjoy coding! 🤓


## How to build
In order to build a new package, a new version can be specified in the [VERSION.txt](/VERSION.txt) file. Then, the package can be created by executing:
```
ofpost-build
``` 


## Examples
Many other examples can be found inside [/test](/test) folder! 🌈

![slice](/test/postProcessing/VelocitySlice/zNormalPlane_U_mag_0.5.png)
![residuals](/test/postProcessing/Residuals/residuals.png)
