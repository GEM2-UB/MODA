# MLcool to use Molecular Orbital Decomposition and Aggregation (MODA)

## Instalation:
- **Create a specific conda environment (for testing purposes):**
    1) Navigate to the `scripts` folder, where the bash script `install.sh` is present.
    2) Type the command `chmod +x install.sh` and execute `source ./install.sh`.
    3) Make sure you are in the correct env.: `conda activate test_moda` every time you use it.ç


- **Install in your usual conda environment (via pip):**
    1) Navigate to the `scripts` folder.
    2) Find the `requirements.txt` file.
    3) Make sure you are in the right conda env.: `conda activate myenv`
    4) Execute `pip install -r requirements.txt`.
    5) Compile the Fortran95 files `cd ./MLcool/descriptors` and type `python -m numpy.f2py -c optimized_kernels.f95 -m optimized_kernels`

## Dataset Overview (DB):
This repository contains a comprehensive dataset documenting the structures and magnetic exchange couplings, denoted as J<sub>AB</sub>, across all datasets considered in this project. The data is neatly organized according to the respective datasets in a hierarchical folder structure for easy navigation and access.

## Folder Structure
### `./OTHER/` 
The `./OTHER/` directory consolidates both the THIL and PHYL datasets.
For individual structural data, browse through the `./OTHER/XYZ/XXX/` directory where you'll find conformers stored in .xyz format. Each conformer is named as `stepN.xyz`, where 'N' represents the angle &theta; associated with each conformer in the rigid body rotational scan.

Magnetic exchange coupling data (J<sub>AB</sub>) for this dataset can be located within the `./OTHER/couplings/` directory. The respective .dat files are labeled as `PHYL.dat` and `THIL.dat`.

### `./TTTA/` 

The `./TTTA/` directory is reserved for the TTTA dataset and follows a structure similar to the `./OTHER/` directory.

In the `./TTTA/XYZ/` directory, you will find subfolders named `HT-XXX-DACB`. Here, 'XXX' corresponds to the temperature value (either 300 or 250), while 'A' and 'B' indicate specific indices denoting the column (C) and dimer (D) positions. Each of these folders contains .xyz files named `STEPN.xyz`, where 'N' acts as a placeholder identifying each sample during the AIMD simulation.

For magnetic exchange coupling data (J<sub>AB</sub>) related to this dataset, navigate to the `./TTTA/couplings/` directory. This directory houses `HT-XXX-DACB.dat` files, each corresponding to a file in the `./TTTA/XYZ/` directory.

## Usage

Find the files in jupyter notebook (`MODA_tutorial.ipynb`) or in pdf format (`MODA_tutorial.pdf`) with a tutorial to an end-to-end case study showing how to use MODA.

