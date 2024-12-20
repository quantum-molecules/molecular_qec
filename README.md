# Source code for: Strategies for implementing quantum error correction in molecular rotation

## GENERAL INFORMATION 

This source code contains the simulations used to generate the data and findings in the publication: Strategies for implementing quantum error correction in molecular rotation (pre-print arXiv: 
https://doi.org/10.48550/arXiv.2405.02236)

This code was modified from the version used to generate the data in manuscript in so far as to upgrade it to run on QuTiP version 5.0 (it was originally executed on version 4.7).

## DATA & FILE OVERVIEW 

File List: 
1. run_dissipative_dynamics.py - python script for performing dissipative quantum error correction simulations on an instance of CounterSymmetricCode (generates Qobj files)
2. run_sequence_basis_exact.py - python script to simulate performance of a single round of exact sequential quantum error correction. 
3. run_sequence_basis_approximate.py - python script to simulate performance of a single round of approximate sequential quantum error correction.
4. run_sequence_decay_exact.py- python script to simulate performance of multiple rounds of exact sequential quantum error correction.
5. run_sequence_decay_approximate.py- python script to simulate performance of multiple rounds of approximate sequential quantum error correction.
6. plot_dissipation_dynamics.ipynb - Jupyter notebook for generating figures related to dissipative quantum error correction displayed in publication from the Qobj files (generates PDF files)
7. plot_sequential.py - python script for generating the figure related to multiple rounds of quantum error correction.
8. rot_qec_counter.py - python module containing CounterSymmetricCode class and associated methods for simulations and plotting
9. requirements.txt - text file containing required python packages


HOW TO PERFORM SIMULATIONS AND PLOT RESULTS: (detailed instructions for Windows)
1. Clone this repository locally and change directory to its location
2. Create virtual environment (python -m venv venv)
3. Activate virtual environment (.\venv\Scripts\activate)
4. Install required packages (pip install -r requirements.txt)
5. Execute run_dissipative_dynamics.py
6. Run all in figure_generator.ipynb
7. Execute run_sequence_basis_exact.py, run_sequence_basis_approximate.py, run_sequence_basis_exact.py, run_sequence_basis_approximate.py
8. Execute plot_sequential.py

## SHARING/ACCESS INFORMATION

This repository is under the BSD 3 clause license

Links to publication that utilize this reposository: 
https://doi.org/10.48550/arXiv.2405.02236

Author/Principal Investigator Information
Name: Philipp Schindler
ORCID: https://orcid.org/0000-0002-9461-9650
Email: philipp.schindler@uibk.ac.at

First Author Information
Name: Brandon J. Furey
ORCID: https://orcid.org/0000-0001-7535-1874
mail: brandon.furey@uibk.ac.at

## Acknowledgements 

Information about funding sources that supported this work: 
- FWF 1000 Ideas project TAI-798 "Molecular Quantum Error Correction"
- ERC Horizon 2020 project ERC-2020-STG 948893 "Quantum characterization and control of single molecules"
