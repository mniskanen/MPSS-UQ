# MPSS-UQ
A Python package to carry out Mobility Particle Size Spectrometer (MPSS) inversion in the Bayesian framework, which provides a natural way to quantify uncertainty. In particular, this package provides a method to model uncertainties in a bipolar charger and propagate those uncertainties to the particle size distribution estimates.

Installing
---
The easiest way to install MPSS-UQ is to create a fresh [conda](https://docs.anaconda.com/miniconda/) environment (python>=3.9 required), clone (or download) this repository and install using pip (note the dot after pip install):
```
conda create -n myenv python=3.11
conda activate myenv
git clone https://github.com/mniskanen/MPSS-UQ.git
cd MPSS-UQ
pip install .
```

Usage
---
In the ``examples/``-folder you will find a script ``invert_simulated_data.py`` which shows the basic workflow for inverting a single MPSS measurement. The script creates simulated DMPS data, inverts it, and compares the inverted result to the true size distribution.
An example of inverting a batch of real measurements is given in ``examples/realdata_UEF_test/``. In this case, we have around 8 days of DMPS measurements with a measured range of 10 to 800 nm covered every 5 minutes.

This is a work in progress so things may change without notice.

License
---

Copyright 2024-2025 Matti Niskanen.

MPSS-UQ is made available under the GPLv3 License. For details see the LICENSE file.
