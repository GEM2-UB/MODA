#!/bin/bash

# Activate the conda environment
conda create test_moda
conda activate test_moda

# Install packages using conda
conda install -c anaconda scikit-learn numpy matplotlib pandas scipy joblib
conda install -c conda-forge pyscf ase nglview
conda install -c conda-forge dscribe

cd ./MLcool/descriptors
python -m numpy.f2py -c optimized_kernels.f95 -m optimized_kernels

# Enable nglview in Jupyter
jupyter-nbextension enable nglview --py --sys-prefix
