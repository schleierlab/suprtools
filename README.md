suprtools: analysis for millimeter-wave cavities
================================================

`suprtools` (**Su**perconducting **P**ost-paraxial **R**esonator tools)
is a data analysis library for
the characterization of millimeter-wave Fabry–Pérot cavities
operating beyond the paraxial limit.


Dependencies
------------

suprtools should work with Python 3.12+. It does _not_ work for Python ≤3.9.

Installation requires the following dependencies
for basic scientific computing:
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scipy](https://www.scipy.org/)
- [lmfit](https://lmfit.github.io/)

For specialized tasks:
- [h5py](https://www.h5py.org/) for interfacing with the HDF5 files used to store data
- [gmsh](https://gmsh.info/) for interfacing with finite-element simulation outputs from [small_fem](https://gitlab.onelab.info/gmsh/small_fem)
- [arc](https://arc-alkali-rydberg-calculator.readthedocs.io/) for Rydberg atom properties
- [openfermion](https://quantumai.google/openfermion) to handle ladder operator algebras
- [scikit-rf](https://scikit-rf.readthedocs.io/) for manipulating frequency-domain rf data

Dependencies that we can maybe remove:
- [scikit-image](https://scikit-image.org/) for phase unwinding

Optionally:
- [jinja](https://jinja.palletsprojects.com/) for generating cryogenic thermometer calibration reports


Installation
------------

Editable install is recommended via pip:

    cd PATH/TO/REPO
    pip install -e .

For features like VSCode IntelliSense to work with this code,
perform a "strict" editable installation

    pip install -e . --config-settings editable_mode=strict


Citing
------

If this software provdes integral to a scientific publication,
please cite [this arXiv manuscript](https://arxiv.org/abs/2506.05804),
in support of which this software was developed.


Testing
-------

Testing requires [pytest](https://docs.pytest.org/).
