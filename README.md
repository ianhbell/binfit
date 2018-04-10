# binfit
The python code for fitting interaction parameters for binary mixtures

From the paper: IH Bell, EW Lemmon, "Automatic Fitting of Binary Interaction Parameters for Multi-fluid Helmholtz-Energy-Explicit Mixture Models", Journal of Chemical & Engineering Data 61 (11), 3752-3760, http://dx.doi.org/10.1021/acs.jced.6b00257 .  If you use this code, please cite that paper.

# Getting Started

You will need some dependencies, most of which can be handled with the Anaconda package installer (https://www.continuum.io/downloads)

1. NIST REFPROP (install as usual)
2. Some other python packages (see below)

Using the conda package manager, you can create a self-contained environment and populate it with the things you need and run the default fit:

    > conda create -n py3 python==3.5 pandas numpy scipy
    > activate py3
    > pip install deap CoolProp xlrd ctREFPROP
    > python binary_fitter.py