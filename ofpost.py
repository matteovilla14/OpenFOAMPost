'''
Load and post-process OpenFOAM simulations. \\
.vtk files will be saved as .png through PyVista. \\
.dat files will be plotted as .png through matplotlib.

Maintainer: TheBusyDev <https://github.com/TheBusyDev/>
'''
import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from io import StringIO
from typing import Generator



# ------------------ INPUT ARGUMENTS ------------------
SUPPORTED_EXTENSIONS = [
    '.png',
    '.jpg',
    '.svg',
    '.eps',
    '.pdf',
]

parser = argparse.ArgumentParser(prog='ofpost',
                                 description='A powerful tool to to post-process OpenFOAM simulations.',
                                 allow_abbrev=False,
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-c', '--case',
                    type=str,
                    choices=['2D', '3D'],
                    default='3D',
                    required=False,
                    help="select case type ('2D' or '3D'). Default: '3D'\n\n")

parser.add_argument('-f', '--format',
                    type=str,
                    choices=SUPPORTED_EXTENSIONS,
                    default=SUPPORTED_EXTENSIONS[0],
                    required=False,
                    help=f"select file format. Default: '{SUPPORTED_EXTENSIONS[0]}'\n\n")

parser.add_argument('-s', '--steady',
                    type=str,
                    choices=['yes', 'no'],
                    default='no',
                    required=False,
                    help="set steady-state case.\nDefault: 'no'\n\n")

args = parser.parse_args()

CASE_TYPE = args.case
EXTENSION = args.format
IS_STEADY = (args.steady == 'yes')



# ------------------ CONSTANTS ------------------
COMPONENTS = ['_x', '_y', '_z'] # all possible arrays components

FORCE_LABEL = 'F'
MOMENT_LABEL = 'M'

UNITS_OF_MEASURE = {
    'p': 'Pa',  # pressure - NOTE: it changes when dealing with incompressible cases
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

if CASE_TYPE == '2D':
    UNITS_OF_MEASURE['F'] = 'N/m'
    UNITS_OF_MEASURE['M'] = 'N*m/m'

if IS_STEADY:    
    UNITS_OF_MEASURE['Time'] = ''

VTK_FILE = r'.*\.vtk'           # .vtk file
CLOUD_FILE = r'cloud_.*\.vtk'
RES_FILE = r'residuals\.dat'    # residuals file
WALL_SHEAR_FILE = r'wallShearStress\.dat' # wallShearStress.dat file
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
    'n_colors': 255, # number of color levels for colormap
    # 'show_edges': True, # uncomment to show the underlying mesh
    # 'edge_color': [200]*3,
    # 'line_width': 2
}

PLOTTER_OPTIONS = {
    'window_size': [1000, 500],
    'background_color': 'white',
}

CAMERA_SCALING = 0.8



# ------------------ MATPLOTLIB OPTIONS ------------------
FIGURE_ARGS = {
    # 'figsize': [8, 6],
    'dpi': 250
}



# ------------------ FUNCTIONS ------------------
def find_files(pattern: str, exceptions: list[str]=[]) -> Generator[str, None, None]:
    '''
    Look for files based on specified pattern recursively. \\
    'pattern' is treated as a regular expression. \\
    'exceptions' is a list of regular expressions. \\
    Return files location as generator. \\
    Print error message if no file is found.
    '''
    pattern = re.compile(pattern)
    exceptions = [re.compile(exc) for exc in exceptions]
    is_found = False
    print(f'\nLooking for {pattern.pattern} files...')

    for root, _, files in os.walk('.', topdown=True):
        for file in files:
            # skip other files
            if not pattern.fullmatch(file):
                continue

            # skip exceptions
            if any([exc.fullmatch(file) for exc in exceptions]):
                continue
            
            # yield filepath
            filepath = os.path.join(root, file)
            is_found = True
            yield filepath

    if not is_found:
        print(f'No {pattern.pattern} file found.')


def get_output_filepath(filepath: str, filesuffix: str='') -> tuple[str, str, str, str]:
    '''
    Return output filepath (with extension), input filename, timestep and output directory name. \\
    If 'filesuffix' is provided, then files will be generated with the specificed suffix.
    '''
    # get timestep and output path based on OpenFOAM convention
    # (under postProcessing folder)
    filename = os.path.basename(filepath)
    filename = filename.split('.')[0:-1] # remove file extension
    filename = ''.join(filename)

    path = os.path.dirname(filepath)
    timestep = os.path.basename(path)

    try:
        float(timestep) # verify that timestep is really a float
    except ValueError:
        timestep = '0'

    # create output file path
    outpath = os.path.dirname(path) # output path
    outdirname = os.path.basename(outpath) # output directory name
    outfilename = filename # output filename

    if filesuffix != '':
        # remove illegal characters from filesuffix
        filesuffix = filesuffix.replace('/', '')
        filesuffix = filesuffix.replace('\\', '')
        filesuffix = filesuffix.replace(' ', '')
        # add filesuffix to output filename
        outfilename += '_' + filesuffix

    if timestep != '0':
        outfilename += '_' + timestep # add timestep to output filename
    
    outfilename += EXTENSION # add extension to output filename
    outfilepath = os.path.join(outpath, outfilename) # output file path
    print(f'Output file: {outfilepath}')

    return outfilepath, filename, timestep, outdirname


def get_units(array_name: str) -> str:
    '''
    Get units of measurement based on input array. \\
    Return empty string if array is not found.
    ''' 
    # detect units of measurement
    try:
        units = ' [' + UNITS_OF_MEASURE[array_name] + ']'
        return units
    except KeyError:
        pass

    # try to extract array_name
    match = re.search(r'\((.*?)\)', array_name)

    if match != None:
        array_name = match.groups()[0]

        # try again to detect units of measurement
        try:
            units = ' [' + UNITS_OF_MEASURE[array_name] + ']'
            return units
        except KeyError:
            pass

    # remove extension from components
    for comp in COMPONENTS:
        if array_name.endswith(comp):
            array_name = array_name.removesuffix(comp)
            break
    
    # try again to detect units of measurement
    try:
        units = ' [' + UNITS_OF_MEASURE[array_name] + ']'
    except KeyError:
        units = ''

    return units


def adjust_camera(plotter: pv.Plotter) -> None:
    '''
    Try to infer slice normal direction and adjust plotter camera position. \\
    If normal direction is not computed correctly \\
    or slice is not aligned with x, y or z direction, then reset camera.
    '''
    mesh = plotter.mesh

    # try to infer slice normal direction (slice has zero thickness in normal direction)
    bounds = np.array(mesh.bounds)
    delta_bounds = np.abs(bounds[1::2] -  bounds[0:-1:2])
    normal_idx, = np.where(delta_bounds < 1e-16) # get zero-thickness direction
    
    # return None if normal is not found correctly
    if len(normal_idx) != 1:
        plotter.reset_camera()
        return
    
    # generate normal vector
    normal = np.zeros(3)
    normal[normal_idx] = 1

    # set up the camera position and focal point
    camera_position = mesh.center + normal * CAMERA_SCALING
    focal_point = mesh.center

    # compute camera view-up orientation
    delta_bounds[normal_idx] = np.inf # needed to avoid the normal direction as view-up direction
    min_bound = np.min(delta_bounds)
    min_indices, = np.where(delta_bounds == min_bound)
    view_up_idx = min_indices[-1] # get view-up direction
    view_up = np.zeros(3)
    view_up[view_up_idx] = 1 if view_up_idx != 2 else -1

    plotter.camera_position = [
        camera_position,
        focal_point,
        view_up
    ]


def vtk2image(filepath: str) -> None:
    '''
    Load .vtk file and convert it to image format as specified by the user.
    '''
    # load VTK file and get array names contained inside it
    mesh = pv.read(filepath)

    if len(mesh.cell_data) > 0:
        print('Loading cell data...')
        data = mesh.cell_data # load cell data as preferred option
    elif len(mesh.point_data) > 0:
        print('Loading point data...')
        data = mesh.point_data # load point data as alternative
    else:
        print('ERROR: empty file.')
        return
    
    array_names = data.keys().copy()
    colormaps = COLORMAPS.copy()

    # loop around all the arrays found in mesh
    for array_name in array_names:
        array = data[array_name]

        # remove empty arrays
        if len(array) == 0:
            data.pop(array_name)
            continue

        # detect units of measurement
        units = get_units(array_name)
        
        # detect 3D arrays
        if array.shape[-1] == 3:
            # split arrays in their components 
            for index, comp in enumerate(COMPONENTS):
                new_name = array_name + comp + units
                data[new_name] = array[:, index]
                colormaps[new_name] = colormaps[array_name] # add entry to colormap

            # rename array to indicate its magnitude
            new_name = array_name + 'mag' + units
        else:
            # add units of measurements to the array
            new_name = array_name + units
        
        # rename array
        data[new_name] = data.pop(array_name)

        # add entry to colormap
        try:
            colormaps[new_name] = colormaps.pop(array_name)
        except KeyError:
            colormaps[new_name] = DEFAULT_COLORMAP
    
    # create a new plotter for pyvista, load mesh and adjust camera
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh)
    adjust_camera(plotter)
    plotter.show_axes()
    plotter.clear()

    # adjust plotter options
    for key, value in PLOTTER_OPTIONS.items():
        setattr(plotter, key, value)

    # loop around the modified arrays
    for array_name in data.keys():
        # plot array and set plot properties
        plotter.add_mesh(mesh,
                         scalars=array_name,
                         cmap=colormaps[array_name],
                         scalar_bar_args=SCALAR_BAR_ARGS,
                         **MESH_ARGS)

        # remove units from array name
        array_name = re.sub(r'\[.*?\]', '', array_name)
        array_name = re.sub(r'\s+', '', array_name)

        # create output directory and save array as png
        outfilepath, *_ = get_output_filepath(filepath, filesuffix=array_name)

        try:
            plotter.screenshot(outfilepath)
        except ValueError:
            plotter.save_graphic(outfilepath)
        
        # clear plotter before starting a new loop
        plotter.clear()
    
    plotter.close() # close plotter


def cloud2png(filepath: str) -> None:
    '''
    Load cloud*.vtk file and convert it to .png.
    '''
    pass # TODO: implement lagrangian particle tracking --> REMOVE


def read_labels(filepath: str) -> list[str]:
    '''
    Get labels from file (i.e. the last comment line).
    '''
    with open(filepath, 'r') as file:
        for line in file:
            # skip comment lines
            if line == '' or line[0] != '#':
                break

            orig_labels = line
    
    try:
        # manipulate labels
        orig_labels = orig_labels.strip()
        orig_labels = orig_labels.removeprefix('#')
        orig_labels = orig_labels.split() # original version of labels
    except:
        return [] # return empty list if any error occured

    # create labels list
    labels = []

    for label in orig_labels:
        # do not append units of measurement
        if not re.match(r'\[.*\]', label):
            labels.append(label)
        else:
            labels[-1] += f' {label}' # append units at the end of the latest label

    return labels


def plot_data(df: pd.DataFrame, filepath: str, semilogy: bool=False, append_units: bool=True, filesuffix: str='') -> None:
    '''
    Plot data from .dat files and save figure. \\
    Receive pandas DataFrame and source filepath as input.
    '''
    # create new plot
    x = df.iloc[:,0]

    if IS_STEADY and x.name == 'Time':
        x.name = 'Iterations'

    fig = plt.figure(**FIGURE_ARGS)

    # loop around DataFrame
    for label, y in df.iloc[:, 1:].items():
        # append units of measurement
        if append_units:
            label += get_units(label)

        # plot data from DataFrame
        if not semilogy:
            plt.plot(x, y, label=label)
        else:
            plt.semilogy(x, y, label=label)

    outfilepath, filename, timestep, outdirname = get_output_filepath(filepath, filesuffix)

    # set plot title
    title = outdirname

    if timestep != '0':
        title += '@' + timestep + UNITS_OF_MEASURE['Time'] # add timestep to plot title
        title += ' (' + filename + ')' # add filename info to plot title

    # set plot xlabel
    xlabel = x.name + get_units(x.name)
    
    # set plot properties
    plt.title(title)
    plt.xlabel(xlabel)
    plt.grid()
    plt.legend()

    # save figure
    plt.savefig(outfilepath)
    plt.close(fig) # close figure once it's saved


def read_dat(filepath: str, semilogy: bool=False, append_units: bool=True) -> None:
    '''
    Read data from .dat files \\
    (for 'forces.dat' files, refer to 'read_forces' function).
    '''
    # initialize labels
    labels = read_labels(filepath)

    # retrive data from yPlus.dat file
    try:
        data = pd.read_csv(filepath, 
                           comment='#',
                           delimiter=r'\t+|\s+',
                           engine='python',
                           names=labels) # set labels
    except Exception as e:
        print(f'ERROR: unable to load {filepath}:')
        print(e)
        return
    
    # plot data and save png
    if 'patch' not in data.columns:
        plot_data(data, filepath, semilogy, append_units)
        return

    # get patch and time DataFrames
    patches = data['patch'].drop_duplicates()
    time = data['Time'].drop_duplicates()
    time.reset_index(drop=True, inplace=True)

    # define field name
    field = os.path.basename(filepath)
    field = field.removesuffix('.dat')

    # find columns
    target_columns = ['min', 'max', 'average']
    old_columns = []

    for col in data.columns:
        if any([col.startswith(tc) for tc in target_columns]):
            old_columns.append(col)
    
    new_columns = [f'{field} {col}' for col in old_columns]

    # rename columns inside DataFrame
    columns = {old_col: new_col
               for old_col, new_col in zip(old_columns, new_columns)}
    data.rename(columns=columns, inplace=True)

    # loop around all the columns
    for col in data.columns:
        # skip useless columns
        if col in ['Time', 'patch']:
            continue

        new_data = time # initialize new DataFrame

        # extract patch data and rename columns for each patch
        for patch in patches:
            # extract patch data
            patch_data = data.loc[data['patch'] == patch]
            patch_data = patch_data[col]

            # insert patch name to column name
            patch_data.name = f'{patch} {patch_data.name}'
            patch_data.reset_index(drop=True, inplace=True)

            # concatenate new data
            new_data = pd.concat([new_data, patch_data], axis=1) 

        # plot selected data
        plot_data(new_data, filepath, semilogy, append_units, filesuffix=col)


def read_forces(filepath: str) -> None:
    '''
    Read data from 'forces.dat' files and save plot.
    '''
    # open force file
    with open(filepath, 'r') as file:
        content = file.read()
    
    # get contributions of each force 
    # (up to 3 contributions: pressure, viscosity, porosity)
    contribs = re.search(r'forces\((.*?)\)', content)

    try:
        n_contribs = len(contribs.group(1).split()) # number of contributions
    except:
        n_contribs = 0

    # remove all the bracket
    content = content.replace('(', ' ')
    content = content.replace(')', ' ')
    dummy_file = StringIO(content)

    try:
        data = pd.read_csv(dummy_file,
                           comment='#', 
                           delimiter=r'\t+|\s+',
                           engine='python')
    except Exception as e:
        print(f'ERROR: unable to load {filepath}:')
        print(e)
        return

    def sum_contribs(start_index: int, label: str, n_contribs: int) -> pd.DataFrame:
        # initialize DataFrame and save Time to DataFrame
        df = pd.DataFrame()
        df['Time'] = data.iloc[:,0]

        # sum contributions from pressure, viscosity and porosity
        indices = range(start_index, start_index+3)
        labels = [label + comp for comp in COMPONENTS]

        for index, label in zip(indices, labels):
            sum_axes = [index + 3*n for n in range(n_contribs)] # get axes to be summed
            df[label] = data.iloc[:, sum_axes].sum(axis=1)
        
        return df

    # get forces and save plot
    start_index = 1
    forces = sum_contribs(start_index, FORCE_LABEL, n_contribs)
    plot_data(forces, filepath, filesuffix='forces')

    # get moments and save plot
    start_index = 1 + 3*n_contribs
    moments = sum_contribs(start_index, MOMENT_LABEL, n_contribs)
    plot_data(moments, filepath, filesuffix='moments')



# ------------------ MAIN PROGRAM ------------------
def main() -> None:
    # analyze .vtk files
    for vtk_file in find_files(VTK_FILE, exceptions=[CLOUD_FILE]):
        print(f'\nProcessing {vtk_file}...')
        vtk2image(vtk_file)

    # analyze cloud files for lagrangian particle tracking
    for cloud_file in find_files(CLOUD_FILE):
        print(f'\nProcessing {cloud_file}...')
        cloud2png(cloud_file)
    
    # analyze .dat and .xy files
    for res_file in find_files(RES_FILE):
        print(f'\nProcessing {res_file}...')
        read_dat(res_file, semilogy=True, append_units=False)

    for dat_file in find_files(DAT_FILE, exceptions=[RES_FILE, WALL_SHEAR_FILE, FORCE_FILE]):
        print(f'\nProcessing {dat_file}...')
        read_dat(dat_file)
    
    for xy_file in find_files(XY_FILE):
        print(f'\nProcessing {xy_file}...')
        read_dat(xy_file)

    for force_file in find_files(FORCE_FILE):
        print(f'\nProcessing {force_file}...')
        read_forces(force_file)

    sys.exit(0)


if __name__ == '__main__':
    main()
