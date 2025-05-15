'''
LOAD CONSTANTS AND OPTIONS.
'''
import sys
import argparse
from pathlib import Path
from typing import Callable
from matplotlib import colormaps as mat_colormaps
from matplotlib.colors import is_color_like, CSS4_COLORS


class opt:
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
        'clim': None,           # range values to show the colormap
        'n_colors': 256,        # number of color levels for colormap
        'show_edges': False,    # show the underlying mesh
        'edge_color': [200]*3,  # underlying mesh color
        'line_width': 1         # underlying mesh line width
    }

    plotter_options = {
        'background_color': 'white',
        'window_size': [1000, 500]
    }

    camera_options = {
        'normal': +1,
        'view_up': +1,
        'focal_point': None,
        'rotate': False,
        'zoom': 0.95
    }


    # ------------------ MATPLOTLIB OPTIONS ------------------
    figure_args = {
        # 'figsize': [8, 6],
        'dpi': 250
    }


    # ------------------ PARSE USER CUSTOM OPTIONS ------------------
    @staticmethod
    def parse_arguments() -> None:
        '''
        Static method to parse user input arguments and change default options.
        '''
        yesno_choices = ['yes', 'no']

        def bool2yesno(bool_var: bool) -> str:
            return ('yes' if bool_var else 'no')
        
        def yesno2bool(str_var: str) -> bool:
            return (str_var == 'yes')
        
        def positive(required_type: Callable) -> Callable:
            def wrapper(input_value: str)-> required_type: # type: ignore
                try:
                    output_value = required_type(input_value) # convert to the required type
                except ValueError:
                    raise argparse.ArgumentTypeError(f"{input_value} is not a valid entry!")

                if output_value <= 0: # check if output value is positive
                    raise argparse.ArgumentTypeError(f"{input_value} is not positive!")

                return output_value

            return wrapper

        positive_int = positive(int)
        positive_float = positive(float)

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
                            help='Paths where post-processing files will be searched recursively')

        # user custom options
        default_2D = bool2yesno(opt.is_2D)

        parser.add_argument('--2D',
                            type=str,
                            choices=yesno_choices,
                            default=default_2D,
                            required=False,
                            help=f"Select case type. Default: {default_2D}\n\n")
        
        default_background = opt.plotter_options['background_color']
        
        parser.add_argument('-b', '--background',
                            type=str,
                            metavar='COLOR',
                            default=default_background,
                            required=False,
                            help=f"Select background color. Default: {default_background}\n\n")

        default_clim = opt.mesh_args['clim']

        parser.add_argument('--clim', 
                            type=float, 
                            nargs=2,
                            metavar=('VMIN', 'VMAX'),
                            default=default_clim,
                            required=False,
                            help="Colormap range, with minimum and maximum values.\n"
                                 "Default: adaptive selection, based on the plotted quantity.\n\n")

        default_cmap = None

        parser.add_argument('--cmap',
                            type=str,
                            default=default_cmap,
                            required=False,
                            help="Select colormap.\n"
                                 "If not specified, colormaps will be automatically selected.\n"
                                 "Refer to matplotlib website to choose the colormap properly.\n\n")

        default_flip_normal = bool2yesno(opt.camera_options['normal'] == -1)

        parser.add_argument('--flip-normal',
                            type=str,
                            choices=yesno_choices,
                            default=default_flip_normal,
                            required=False,
                            help=f"Flip normal direction when plotting slices. Default: {default_flip_normal}\n\n")

        default_flip_view_up = bool2yesno(opt.camera_options['view_up'] == -1)

        parser.add_argument('--flip-view-up',
                            type=str,
                            choices=yesno_choices,
                            default=default_flip_view_up,
                            required=False,
                            help=f"Flip view-up direction when plotting slices. Default: {default_flip_view_up}\n\n")

        default_focal_point = opt.camera_options['focal_point']

        parser.add_argument('--focal-point',
                            type=float,
                            nargs=3,
                            metavar=('X', 'Y', 'Z'),
                            default=default_focal_point,
                            required=False,
                            help="Set focal point when plotting slices.\n"
                                 "Default: take focal point at the middle of the mesh\n\n")

        parser.add_argument('-f', '--format',
                            type=str,
                            choices=opt.SUPPORTED_EXTENSIONS,
                            default=opt.extension,
                            required=False,
                            help=f"Select file format. Default: {opt.extension}\n\n")

        default_incomp = bool2yesno(opt.is_incomp)

        parser.add_argument('-i', '--incomp',
                            type=str,
                            choices=yesno_choices,
                            default=default_incomp,
                            required=False,
                            help=f"Set incompressible case. Default: {default_incomp}\n\n")
        
        default_n_colors = opt.mesh_args['n_colors']

        parser.add_argument('-n', '--n-colors',
                            type=positive_int,
                            metavar='N',
                            default=default_n_colors,
                            required=False,
                            help=f"Set number of colors used to display scalars. Default: {default_n_colors}\n\n")
        
        default_rotate = bool2yesno(opt.camera_options['rotate'])

        parser.add_argument('-r', '--rotate',
                            type=str,
                            choices=yesno_choices,
                            default=default_rotate,
                            required=False,
                            help=f"Rotate slice by 90 degrees. Default: {default_rotate}\n\n")
        
        default_show_edges = bool2yesno(opt.mesh_args['show_edges'])

        parser.add_argument('--show-edges',
                            type=str,
                            choices=yesno_choices,
                            default=default_show_edges,
                            required=False,
                            help=f"Show underlying mesh. Default: {default_show_edges}\n\n")
        
        default_steady = bool2yesno(opt.is_steady)

        parser.add_argument('-s', '--steady',
                            type=str,
                            choices=yesno_choices,
                            default=default_steady,
                            required=False,
                            help=f"Set steady-state case. Default: {default_steady}\n\n")
        
        default_window_size = opt.plotter_options['window_size']

        parser.add_argument('-w', '--window-size',
                            type=positive_int,
                            nargs=2,
                            metavar=('WIDTH', 'HEIGHT'),
                            default=default_window_size,
                            required=False,
                            help=f"Set window size. Default: {default_window_size[0]} {default_window_size[1]} \n\n")

        default_zoom = opt.camera_options['zoom']

        parser.add_argument('-z', '--zoom',
                            type=positive_float,
                            default=default_zoom,
                            required=False,
                            help="Set camera zoom. It must be a positive float.\n"
                                 "A value greater than 1 is a zoom-in, a value less than 1 is a zoom-out.\n"
                                 f"Default: {default_zoom}\n\n")

        # parse arguments
        args = parser.parse_args()

        # ------------------ positional arguments ------------------
        # check if path actually exists and append them to 'paths'
        for path in args.paths:
            if not path.exists():
                print(f'ERROR: {path} directory does not exist...')
                sys.exit(1)
            else:
                opt.paths.append(path.absolute())

        opt.is_2D = yesno2bool(getattr(args, '2D'))
        opt.is_incomp = yesno2bool(args.incomp)
        opt.is_steady = yesno2bool(args.steady)

        # ------------------ generic options ------------------
        if opt.is_2D:
            opt.units_of_measure['F'] = 'N/m'
            opt.units_of_measure['M'] = 'N*m/m'

        if opt.is_incomp:
            opt.units_of_measure['p'] = 'm^2/s^2' # kinematic pressure is used in incompressible simulations

        if opt.is_steady:
            opt.units_of_measure['Time'] = ''

        opt.extension = args.format # extension to be used to save files

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
            opt.default_colormap = args.cmap
            opt.colormaps = {}

        opt.mesh_args['clim'] = args.clim
        opt.mesh_args['n_colors'] = args.n_colors
        opt.mesh_args['show_edges'] = yesno2bool(args.show_edges)

        # check if background color is a valid entry
        if not is_color_like(args.background):
            print(f'ERROR: {args.background} is not a valid entry!\n'
                  'Here is a list of accepted named colors:\n\n -> ',
                  end='')
            print('\n -> '.join(CSS4_COLORS.keys()))
            print()
            sys.exit(1)
        
        opt.plotter_options['background_color'] = args.background
        opt.plotter_options['window_size'] = args.window_size

        if not yesno2bool(args.flip_normal):
            opt.camera_options['normal'] = +1
        else:
            opt.camera_options['normal'] = -1

        if not yesno2bool(args.flip_view_up):
            opt.camera_options['view_up'] = +1
        else:
            opt.camera_options['view_up'] = -1

        opt.camera_options['focal_point'] = args.focal_point
        opt.camera_options['rotate'] = yesno2bool(args.rotate)
        opt.camera_options['zoom'] = args.zoom

        # place scalar bar verically
        if opt.camera_options['rotate']:
            sba = opt.scalar_bar_args
            sba['vertical'] = True
            sba['width'], sba['height'] = sba['height'], sba['width']
            sba['position_y'] = sba['position_x']
            sba['position_x'] = 0.85