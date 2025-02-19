'''
MAIN PROGRAM.
'''
import sys


# ------------------ IMPORT CONSTANTS ------------------
from ofpost import PATHS, VTK_FILE, CLOUD_FILE, \
                   RES_FILE, DAT_FILE, XY_FILE, FORCE_FILE


# ------------------ IMPORT FUNCTIONS ------------------
from ofpost.lib import find_files, vtk2image, read_dat, read_forces



# ------------------ MAIN PROGRAM ------------------
for path in PATHS:
    # look for files to be analyzed in each path
    # analyze .vtk files
    for vtk_file in find_files(VTK_FILE, path,
                               exceptions=[CLOUD_FILE]):
        vtk2image(vtk_file)

    # analyze .dat and .xy files
    for res_file in find_files(RES_FILE, path):
        read_dat(res_file, semilogy=True, append_units=False)

    for dat_file in find_files(DAT_FILE, path,
                               exceptions=[RES_FILE, FORCE_FILE]):
        read_dat(dat_file)

    for xy_file in find_files(XY_FILE, path):
        read_dat(xy_file)

    for force_file in find_files(FORCE_FILE, path):
        read_forces(force_file)

sys.exit(0)
