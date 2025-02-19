'''
Load and post-process OpenFOAM simulations. \\
.vtk files will be exported as images through PyVista. \\
.dat files will be plotted through matplotlib.

Maintainer: TheBusyDev <https://github.com/TheBusyDev/>
'''
import sys
import argparse
from pathlib import Path
from matplotlib import colormaps



# ------------------ INPUT ARGUMENTS & CONSTANTS ------------------
SUPPORTED_EXTENSIONS = [
    '.png',
    '.jpg',
    '.svg',
    '.eps',
    '.pdf'
]

DEFAULT_BACKGROUND = 'white'
DEFAULT_N_COLORS = 256
DEFAULT_WINDOW_SIZE = [1000, 500]
DEFAULT_ZOOM = 1.75

# argument parser
parser = argparse.ArgumentParser(prog='ofpost',
                                 description='A powerful tool to to post-process OpenFOAM simulations.',
                                 allow_abbrev=False,
                                 formatter_class=argparse.RawTextHelpFormatter)

# positional arguments
parser.add_argument('paths',
                    type=Path,
                    nargs='+',
                    metavar='PATHS',
                    help='paths where post-processing files will be looked for recursively')

# user custom options
parser.add_argument('-b', '--background',
                    type=str,
                    metavar='COLOR',
                    default=DEFAULT_BACKGROUND,
                    required=False,
                    help=f"select background color. Default: {DEFAULT_BACKGROUND}\n\n")

parser.add_argument('-c', '--case',
                    type=str,
                    choices=['2D', '3D'],
                    default='3D',
                    required=False,
                    help="select case type. Default: 3D\n\n")

parser.add_argument('--cmap',
                    type=str,
                    default=None,
                    required=False,
                    help=f"select colormap.\n"
                          "If not specified, colormaps will be automatically selected.\n"
                          "Refer to matplotlib website to choose the colormap properly\n\n")

parser.add_argument('-f', '--format',
                    type=str,
                    choices=SUPPORTED_EXTENSIONS,
                    default=SUPPORTED_EXTENSIONS[0],
                    required=False,
                    help=f"select file format. Default: {SUPPORTED_EXTENSIONS[0]}\n\n")

parser.add_argument('-i', '--incomp',
                    type=str,
                    choices=['yes', 'no'],
                    default='no',
                    required=False,
                    help="set incompressible case. Default: no\n\n")

parser.add_argument('-n', '--n-colors',
                    type=int,
                    metavar='N',
                    default=DEFAULT_N_COLORS,
                    required=False,
                    help=f"set number of colors used to display scalars. Default: {DEFAULT_N_COLORS}\n\n")

parser.add_argument('-s', '--steady',
                    type=str,
                    choices=['yes', 'no'],
                    default='no',
                    required=False,
                    help="set steady-state case. Default: no\n\n")

parser.add_argument('-w', '--window-size',
                    type=int,
                    nargs=2,
                    metavar=('WIDTH', 'HEIGHT'),
                    default=DEFAULT_WINDOW_SIZE,
                    required=False,
                    help=f"set window size. Default: {DEFAULT_WINDOW_SIZE[0]} {DEFAULT_WINDOW_SIZE[1]}\n\n")

parser.add_argument('-z', '--zoom',
                    type=float,
                    default=DEFAULT_ZOOM,
                    required=False,
                    help=f"set camera zoom. Default: {DEFAULT_ZOOM}\n\n")

# parse arguments
args = parser.parse_args()

# positional arguments
PATHS = []

# check if path actually exists and append them to PATHS
for path in args.paths:
    if not path.exists():
        print(f'ERROR: {path} does not exist...')
    else:
        PATHS.append(path)

# user custom options
IS_2D = (args.case == '2D')
IS_INCOMP = (args.incomp == 'yes')
IS_STEADY = (args.steady == 'yes')



# ------------------ CONSTANTS ------------------
EXTENSION = args.format # extension to be used to save files
COMPONENTS_EXT = ['_x', '_y', '_z'] # all possible arrays components
MAGNITUDE_EXT = '_mag' # magnitude extension

FORCE_LABEL = 'F'
MOMENT_LABEL = 'M'

UNITS_OF_MEASURE = {
    'p': 'Pa',  # pressure
    'U': 'm/s', # velocity
    'T': 'K',   # temperature
    'Ma': '-',  # Mach number
    'F': 'N',   # force
    'M': 'N*m', # moment
    'x': 'm',   # x direction
    'y': 'm',   # y direction
    'z': 'm',   # z direction
    'delta': 'm', # film thickness
    'Time': 's' # time
}

if IS_2D:
    UNITS_OF_MEASURE['F'] = 'N/m'
    UNITS_OF_MEASURE['M'] = 'N*m/m'

if IS_INCOMP:
    UNITS_OF_MEASURE['p'] = 'm^2/s^2' # kinematic pressure is used in incompressible simulations

if IS_STEADY:    
    UNITS_OF_MEASURE['Time'] = ''

VTK_FILE = '*.vtk'          # .vtk file
CLOUD_FILE = 'cloud_*.vtk'
RES_FILE = 'residuals.dat'  # residuals file
DAT_FILE = '*.dat'          # .dat file
XY_FILE = '*.xy'            # .xy file
FORCE_FILE = 'forces.dat'   # forces.dat file



# ------------------ PYVISTA OPTONS ------------------
if args.cmap == None:
    DEFAULT_COLORMAP = 'coolwarm'

    COLORMAPS = {
        'p': 'coolwarm',
        'U': 'turbo',
        'T': 'inferno',
        'Ma': 'turbo',
        'C7H16': 'hot',
        'H2': 'hot',
        'O2': 'viridis',
        'N2': 'winter',
        'H2O': 'ocean'
    }
else:
    matplotlib_cmaps = colormaps()

    # check if colormap is valid
    if not args.cmap in matplotlib_cmaps:
        print(f'ERROR: {args.cmap} is not a valid entry!\n'
              'Here is a list of accepted colormaps:\n\n -> ',
              end='')
        print('\n -> '.join(matplotlib_cmaps))
        print()
        sys.exit(1)

    # force to use user defined colormap
    DEFAULT_COLORMAP = args.cmap
    COLORMAPS = {}
    del matplotlib_cmaps

SCALAR_BAR_ARGS = {
    'vertical': False,
    'width': 0.7,
    'height': 0.05,
    'position_x': 0.15,
    'position_y': 0.05,
    'n_labels': 6,
    'title_font_size': 20,
    'label_font_size': 18,
    'font_family': 'times'
}

MESH_ARGS = {
    'n_colors': args.n_colors, # number of color levels for colormap
    # 'show_edges': True, # uncomment to show the underlying mesh
    # 'edge_color': [200]*3,
    # 'line_width': 2
}

PLOTTER_OPTIONS = {
    'window_size': args.window_size,
    'background_color': args.background,
}

CAMERA_ZOOM = args.zoom



# ------------------ MATPLOTLIB OPTIONS ------------------
FIGURE_ARGS = {
    # 'figsize': [8, 6],
    'dpi': 250
}
