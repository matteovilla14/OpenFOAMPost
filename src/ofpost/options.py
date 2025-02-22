'''
LOAD CONSTANTS AND OPTIONS.
'''
import sys
import argparse
from pathlib import Path
from matplotlib import colormaps as mat_colormaps
from matplotlib.colors import is_color_like, CSS4_COLORS



# ------------------ CONSTANTS ------------------
SUPPORTED_EXTENSIONS = [
    '.png',
    '.jpg',
    '.svg',
    '.eps',
    '.pdf'
]

WORKING_PATH = Path.cwd()   # currently working path

VTK_FILE = '*.vtk'          # .vtk file
CLOUD_FILE = 'cloud_*.vtk'  # cloud file
RES_FILE = 'residuals.dat'  # residuals file
DAT_FILE = '*.dat'          # .dat file
XY_FILE = '*.xy'            # .xy file
FORCE_FILE = 'forces.dat'   # forces.dat file

COMPONENTS_EXT = ['_x', '_y', '_z'] # all possible arrays components
MAGNITUDE_EXT = '_mag'              # magnitude extension

FORCE_LABEL = 'F'
MOMENT_LABEL = 'M'



# ------------------ GENERIC OPTIONS ------------------
paths = [] # list provided by the user of paths

is_2D = False       # for 2D simulations
is_incomp = False   # for incompressible simulations
is_steady = False   # for steady simulations

extension = SUPPORTED_EXTENSIONS[0] # extension to be used to save files

units_of_measure = {
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



# ------------------ PYVISTA OPTONS ------------------
default_colormap = 'coolwarm'

colormaps = {
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

scalar_bar_args = {
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

mesh_args = {
    'n_colors': 256,        # number of color levels for colormap
    'show_edges': False,    # show the underlying mesh
    'edge_color': [200]*3,  # underlying mesh color
    'line_width': 1         # underlying mesh line width
}

plotter_options = {
    'background_color': 'white',
    'window_size': [1000, 500]
}

camera_zoom = 1.75



# ------------------ MATPLOTLIB OPTIONS ------------------
figure_args = {
    # 'figsize': [8, 6],
    'dpi': 250
}



# ------------------ PARSE USER CUSTOM OPTIONS ------------------
def parse_options() -> None:
    '''
    Parse user input arguments and change default options.
    '''
    global paths, is_2D, is_incomp, is_steady, extension, units_of_measure
    global default_colormap, colormaps, mesh_args, plotter_options, camera_zoom

    yesno_choices = ['yes', 'no']

    def bool2yesno(bool_var: bool) -> str:
        return ('yes' if bool_var else 'no')
    
    def yesno2bool(str_var: str) -> bool:
        return (str_var == 'yes')
    
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
    default_2D = bool2yesno(is_2D)

    parser.add_argument('--2D',
                        type=str,
                        choices=yesno_choices,
                        default=default_2D,
                        required=False,
                        help=f"select case type. Default: {default_2D}\n\n")
    
    default_background = plotter_options['background_color']
    
    parser.add_argument('-b', '--background',
                        type=str,
                        metavar='COLOR',
                        default=default_background,
                        required=False,
                        help=f"select background color. Default: {default_background}\n\n")

    parser.add_argument('--cmap',
                        type=str,
                        default=None,
                        required=False,
                        help=f"select colormap.\n"
                             "If not specified, colormaps will be automatically selected.\n"
                             "Refer to matplotlib website to choose the colormap properly.\n\n")

    parser.add_argument('-f', '--format',
                        type=str,
                        choices=SUPPORTED_EXTENSIONS,
                        default=extension,
                        required=False,
                        help=f"select file format. Default: {extension}\n\n")
    
    default_incomp = bool2yesno(is_incomp)

    parser.add_argument('-i', '--incomp',
                        type=str,
                        choices=yesno_choices,
                        default=default_incomp,
                        required=False,
                        help=f"set incompressible case. Default: {default_incomp}\n\n")
    
    default_n_colors = mesh_args['n_colors']

    parser.add_argument('-n', '--n-colors',
                        type=int,
                        metavar='N',
                        default=default_n_colors,
                        required=False,
                        help=f"set number of colors used to display scalars. Default: {default_n_colors}\n\n")
    
    default_show_edges = bool2yesno(mesh_args['show_edges'])

    parser.add_argument('--show-edges',
                        type=str,
                        choices=yesno_choices,
                        default=default_show_edges,
                        required=False,
                        help=f"show underlying mesh. Default: {default_show_edges}\n\n")
    
    default_steady = bool2yesno(is_steady)

    parser.add_argument('-s', '--steady',
                        type=str,
                        choices=yesno_choices,
                        default=default_steady,
                        required=False,
                        help=f"set steady-state case. Default: {default_steady}\n\n")
    
    default_window_size = plotter_options['window_size']

    parser.add_argument('-w', '--window-size',
                        type=int,
                        nargs=2,
                        metavar=('WIDTH', 'HEIGHT'),
                        default=default_window_size,
                        required=False,
                        help=f"set window size. Default: {default_window_size[0]} {default_window_size[1]} \n\n")

    parser.add_argument('-z', '--zoom',
                        type=float,
                        default=camera_zoom,
                        required=False,
                        help=f"set camera zoom. Default: {camera_zoom}\n\n")

    # parse arguments
    args = parser.parse_args()

    # ------------------ positional arguments ------------------
    # check if path actually exists and append them to 'paths'
    for path in args.paths:
        if not path.exists():
            print(f'ERROR: {path} directory does not exist...')
            sys.exit(1)
        else:
            paths.append(path.absolute())

    is_2D = yesno2bool(getattr(args, '2D'))
    is_incomp = yesno2bool(args.incomp)
    is_steady = yesno2bool(args.steady)

    # ------------------ generic options ------------------
    if is_2D:
        units_of_measure['F'] = 'N/m'
        units_of_measure['M'] = 'N*m/m'

    if is_incomp:
        units_of_measure['p'] = 'm^2/s^2' # kinematic pressure is used in incompressible simulations

    if is_steady:    
        units_of_measure['Time'] = ''

    extension = args.format # extension to be used to save files

    # ------------------ pyvista optons ------------------
    if args.cmap != None:
        mat_cmaps = mat_colormaps()

        # check if colormap is valid
        if not args.cmap in mat_cmaps:
            print(f'ERROR: {args.cmap} is not a valid entry!\n'
                  'Here is a list of accepted colormaps:\n\n -> ',
                  end='')
            print('\n -> '.join(mat_cmaps))
            print()
            sys.exit(1)

        # force to use user-defined colormap
        default_colormap = args.cmap
        colormaps = {}

    mesh_args['n_colors'] = args.n_colors
    mesh_args['show_edges'] = yesno2bool(args.show_edges)

    # check if background color is a valid entry
    if not is_color_like(args.background):
        print(f'ERROR: {args.background} is not a valid entry!\n'
              'Here is a list of accepted named colors:\n\n -> ',
              end='')
        print('\n -> '.join(CSS4_COLORS.keys()))
        print()
        sys.exit(1)
    
    plotter_options['background_color'] = args.background
    plotter_options['window_size'] = args.window_size

    camera_zoom = args.zoom
