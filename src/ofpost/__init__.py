'''
Load and post-process OpenFOAM simulations. \\
.vtk files will be exported as images through PyVista. \\
.dat files will be plotted through matplotlib.

Maintainer: TheBusyDev <https://github.com/TheBusyDev/>
'''
import argparse



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


parser = argparse.ArgumentParser(prog='ofpost',
                                 description='A powerful tool to to post-process OpenFOAM simulations.',
                                 allow_abbrev=False,
                                 formatter_class=argparse.RawTextHelpFormatter)

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

args = parser.parse_args()

BACKGROUND = args.background
IS_2D = (args.case == '2D')
EXTENSION = args.format
IS_INCOMP = (args.incomp == 'yes')
N_COLORS = args.n_colors
IS_STEADY = (args.steady == 'yes')
WINDOW_SIZE = args.window_size
CAMERA_ZOOM = args.zoom



# ------------------ CONSTANTS ------------------
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

VTK_FILE = r'.*\.vtk'           # .vtk file
CLOUD_FILE = r'cloud_.*\.vtk'
RES_FILE = r'residuals\.dat'    # residuals file
DAT_FILE = r'.*\.dat'           # .dat file
XY_FILE = r'.*\.xy'             # .xy file
FORCE_FILE = r'forces\.dat'     # forces.dat file



# ------------------ PYVISTA OPTONS ------------------
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
    'n_colors': N_COLORS, # number of color levels for colormap
    # 'show_edges': True, # uncomment to show the underlying mesh
    # 'edge_color': [200]*3,
    # 'line_width': 2
}

PLOTTER_OPTIONS = {
    'window_size': WINDOW_SIZE,
    'background_color': BACKGROUND,
}



# ------------------ MATPLOTLIB OPTIONS ------------------
FIGURE_ARGS = {
    # 'figsize': [8, 6],
    'dpi': 250
}
