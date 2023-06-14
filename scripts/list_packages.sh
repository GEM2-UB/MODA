#!/bin/bash

# Activate the conda environment
conda activate moda_test

# List of packages
packages=("pyscf" "scikit-learn" "ase" "nglview" "numpy" "matplotlib" "dscribe" "joblib" "pandas" "scipy")

# Loop through the packages and execute `conda list`
for package in "${packages[@]}"
do
   echo "Checking package: $package"
   conda list $package
done
