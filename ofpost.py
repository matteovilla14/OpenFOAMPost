'''
Load and post-process OpenFOAM simulations. \\
.vtk files will be saved as .svg through PyVista. \\
.dat files will be plotted as .svg through matplotlib.

Maintainer: TheBusyDev <https://github.com/TheBusyDev/>
'''
import os
import re
import sys
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from io import StringIO



# ------------------ CONSTANTS ------------------
CASE_TYPE = '2D' # other option: 3D

ALL_COMPONENTS = ['_x', '_y', '_z'] # all possible arrays components

match CASE_TYPE:
    case '2D': IN_COMPONENTS = ['_x', '_y']       # in-plane components (for 2D case)
    case '3D': IN_COMPONENTS = ['_x', '_y', '_z'] # components 

# out-of-plane components (for 2D case)
OUT_COMPONENTS = [comp for comp in ALL_COMPONENTS
                  if comp not in IN_COMPONENTS]

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
    'Time': 's' # time
}

VTK_FILE = r'.*\.vtk' # .vtk file
CLOUD_FILE = r'cloud_.*\.vtk'
RES_FILE = r'residuals\.dat' # residuals file
XY_FILE = r'.*\.xy' # .cy file
FORCE_FILE = r'forces\.dat' # forces.dat file



# ------------------ PYVISTA OPTONS ------------------
DEFAULT_COLORMAP = 'coolwarm'

COLORMAPS = {
    'p': 'coolwarm',
    'U': 'turbo',
    'T': 'inferno',
    'Ma': 'turbo'
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

PLOTTER_OPTIONS = {
    'window_size': [1000, 500],
    'background_color': 'white',
    'camera_position': 'yx'
}

CAMERA_ZOOM = 2



# ------------------ MATPLOTLIB OPTIONS ------------------
FIGURE_ARGS = {
    'figsize': [8, 6],
    'dpi': 125
}



# ------------------ FUNCTIONS ------------------
def find_files(pattern: str, exceptions: list[str]=[]) -> list[str]:
    '''
    Look for files based on specified pattern recursively. \\
    'pattern' is treated as a regular expression. \\
    'exceptions' is a list of regular expressions. \\
    Return files location as list. \\
    Return an empty list if no file is found.
    '''
    pattern = re.compile(pattern)
    exceptions = [re.compile(exc) for exc in exceptions]
    print(f'\nLooking for {pattern.pattern} files...')
    filepaths = []

    for root, _, files in os.walk('.', topdown=True):
        for file in files:
            # skip other files
            if not pattern.fullmatch(file):
                continue

            # skip exceptions
            if any([exc.fullmatch(file) for exc in exceptions]):
                continue
            
            # append filepath
            filepath = os.path.join(root, file)
            filepaths.append(filepath)

    if filepaths == []:
        print(f'No {pattern.pattern} file found.')
    
    return filepaths


def get_output_filepath(filepath: str, filesuffix: str='') -> tuple[str, str, str, str]:
    '''
    Return output filepath (with .svg extension), input filename, timestep and output directory name. \\
    If 'filesuffix' is provided, then files will be generated with the specificed suffix.
    '''
    # get timestep and output path based on OpenFOAM convention
    # (under postProcessing folder)
    filename = os.path.basename(filepath)
    filename = filename.split('.')[0:-1] # remove file extension
    filename = ''.join(filename)

    path = os.path.dirname(filepath)
    timestep = os.path.basename(path)

    # create output file path
    outpath = os.path.dirname(path) # output path
    outdirname = os.path.basename(outpath) # output directory name
    outfilename = filename # output filename

    if filesuffix != '':
        outfilename += '_' + filesuffix # add filesuffix to output filename

    if timestep != '0':
        outfilename += '_' + timestep # add timestep to output filename
    
    outfilename += '.svg' # add extension to output filename
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

    # remove extension from components
    for comp in ALL_COMPONENTS:
        if array_name.endswith(comp):
            array_name = array_name.removesuffix(comp)
            break
    
    # try again to detect units of measurement
    try:
        units = ' [' + UNITS_OF_MEASURE[array_name] + ']'
    except KeyError:
        units = ''

    return units


def vtk2svg(filepath: str) -> None:
    '''
    Load .vtk file and convert it to svg.
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
            for index, comp in enumerate(IN_COMPONENTS):
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

    # loop around the modified arrays
    for array_name in data.keys():
        # plot array and set plot properties
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh,
                         scalars=array_name,
                         cmap=colormaps[array_name],
                         scalar_bar_args=SCALAR_BAR_ARGS)
        
        # adjust plotter options
        for key, value in PLOTTER_OPTIONS.items():
            setattr(plotter, key, value)

        plotter.zoom_camera(CAMERA_ZOOM)

        # remove units from array name
        array_name = re.sub(r'\[.*?\]', '', array_name)
        array_name = re.sub(r'\s+', '', array_name)

        # create output directory and save array as svg
        outfilepath, *_ = get_output_filepath(filepath, filesuffix=array_name)
        plotter.save_graphic(outfilepath)
        plotter.close()


def cloud2svg(filepath: str) -> None:
    '''
    Load cloud*.vtk file and convert it to svg.
    '''
    pass # TODO: implement lagrangian particle tracking


def read_labels(filepath: str) -> list[str]:
    '''
    Get labels from file (i.e. the last comment line).
    '''
    with open(filepath, 'r') as file:
        for line in file:
            # skip comment lines
            if line == '' or line[0] != '#':
                break

            labels = line
    
    # analyze last line and get labels
    try:
        return labels[1:].split()
    except:
        return []


def plot_data(df: pd.DataFrame, filepath: str, semilogy: bool=False, append_units: bool=True) -> None:
    '''
    Plot data from .dat files and save figure. \\
    Receive pandas DataFrame and source filepath as input.
    '''
    # create new plot
    x = df.iloc[:,0]
    fig, ax = plt.subplots(**FIGURE_ARGS)

    # loop around DataFrame
    for label, y in df.iloc[:, 1:].items():
        # append units of measurement
        if append_units:
            label += get_units(label)

        # plot data from DataFrame
        if not semilogy:
            ax.plot(x, y, label=label)
        else:
            ax.semilogy(x, y, label=label)

    outfilepath, filename, timestep, outdirname = get_output_filepath(filepath)

    # set plot title
    title = outdirname

    if timestep != '0':
        title += '@' + timestep + UNITS_OF_MEASURE['Time'] # add timestep to plot title
        title += ' (' + filename + ')' # add filename info to plot title

    # set plot xlabel
    xlabel = x.name + get_units(x.name)
    
    # set plot properties
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid()
    ax.legend()

    # save figure
    fig.savefig(outfilepath)    
    plt.close(fig) # close figure once it's saved


def read_dat(filepath: str, semilogy: bool=False, append_units: bool=True) -> None:
    '''
    Read data from .dat files \\
    (for 'forces.dat' files, refer to 'read_forces' function).
    '''
    # initialize labels
    labels = read_labels(filepath)

    # retrive data from .dat file
    try:
        data = pd.read_csv(filepath, 
                           comment='#',
                           delimiter=r'\t+|\s+',
                           engine='python',
                           names=labels) # set labels
    except:
        print(f'ERROR: unable to load {filepath}.')
        return
    
    # do not plot out-of-plane component in 2D case
    if CASE_TYPE == '2D':        
        for label in labels:
            for comp in OUT_COMPONENTS:
                if label.endswith(comp):
                    data.drop(label, axis=1, inplace=True) # drop out-of-plane component
                    break                
    
    # plot data and save svg
    plot_data(data, filepath, semilogy, append_units)


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
    except:
        print(f'ERROR: unable to load {filepath}.')
        return

    # save iterations to 'forces' DataFrame
    forces = pd.DataFrame()
    forces['Time'] = data.iloc[:,0]

    def sum_contribs(forces, start_index, labels, n_contribs):
        # sum contributions from pressure, viscosity and porosity
        indices = range(start_index, start_index+3)

        for index, label in zip(indices, labels):
            sum_axes = [index + 3*n for n in range(n_contribs)] # get axes to be summed
            forces[label] = data.iloc[:, sum_axes].sum(axis=1)

    # get forces
    f_labels = [FORCE_LABEL + comp for comp in IN_COMPONENTS]
    start_index = 1
    sum_contribs(forces, start_index, f_labels, n_contribs)

    # get moments
    match CASE_TYPE:
        case '2D': m_labels = [MOMENT_LABEL + comp for comp in OUT_COMPONENTS]
        case '3D': m_labels = [MOMENT_LABEL + comp for comp in IN_COMPONENTS]

    start_index = 1 + 3*n_contribs
    sum_contribs(forces, start_index, m_labels, n_contribs)

    # plot forces and save svg
    plot_data(forces, filepath)

    return forces



# ------------------ MAIN PROGRAM ------------------
def main() -> None:
    # analyze .vtk files
    for vtk_file in find_files(VTK_FILE, exceptions=[CLOUD_FILE]):
        print(f'\nProcessing {vtk_file}...')
        vtk2svg(vtk_file)

    # analyze cloud files for lagrangian particle tracking
    for cloud_file in find_files(CLOUD_FILE):
        print(f'\nProcessing {cloud_file}...')
        cloud2svg(cloud_file)
    
    # analyze .dat and .xy files
    for res_file in find_files(RES_FILE):
        print(f'\nProcessing {res_file}...')
        read_dat(res_file, semilogy=True, append_units=False)
    
    for xy_file in find_files(XY_FILE):
        print(f'\nProcessing {xy_file}...')
        read_dat(xy_file)

    for force_file in find_files(FORCE_FILE):
        print(f'\nProcessing {force_file}...')
        read_forces(force_file)

    sys.exit(0)


if __name__ == '__main__':
    main()
