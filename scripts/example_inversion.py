# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as resources

from MPSS_UQ.particlesizers import DifferentialMobilityParticleSizer
from MPSS_UQ.inversion import (Laplace_approximation,
                                     Laplace_approximation_marginalize,
                                     smoothness_prior
                                     )
from MPSS_UQ.synthetic_data import generate_DMPS_measurement
from MPSS_UQ.plotfunctions import plot_psd, plot_marginalized_psd


'''
This script shows an example on how to use this package for inversion of DMPS data.
In this example we use synthetic data generated with a DMPS model, but using real measurements
is essentially identical to this example, one just needs to replace the first part of this script
with loading the measured data.
'''



# %% Load (generate) data

# Load a DMPS configuration file. This includes basic geometry and flow rates information
# on the DMPS. This info can also be specified here in code.
fname = resources.files('MPSS_UQ.data') / 'DMPS_properties.json'
with open(fname, 'r') as f:
    DMPS_prop = json.load(f)['UEF-A20']

# Set the mobility diameters the DMPS should measure
DMPS_prop['d_m_data'] = np.geomspace(6e-9, 500e-9, num=30)

# Set up some properties of the DMPS used to create measurement data
DMPS_prop_datagen = DMPS_prop.copy()
DMPS_prop_datagen['max_charge'] = 8  # Maximum considered number of multiple charges
DMPS_prop_datagen['charging_model'] = 'LYF-interp'  # Which bipolar charging model to use

# Mobility diameters for the true PSD
DMPS_prop_datagen['d_m'] = np.geomspace(2e-9, 2500e-9, num=500)

# Create the data generating DMPS model
DMPS_datagen = DifferentialMobilityParticleSizer(DMPS_prop_datagen)

# Set ion properties for the measurement
pos_ion_mobility = 1.4e-4
neg_ion_mobility = 1.9e-4
ion_ratio = 1.0
DMPS_datagen.update_charger_ion_properties(pos_ion_mobility, neg_ion_mobility, ion_ratio)

# Create the measurement
measurement = generate_DMPS_measurement(DMPS_datagen)



# %% Inversion (MAP-estimate)

# Set up a DMPS model for inversion (Wiedensohler)
DMPS_prop['max_charge'] = 4
DMPS_prop['charging_model'] = 'Wiedensohler'

# Particle mobility diameters for inversion
DMPS_prop['d_m'] = np.geomspace(6e-9, 2500e-9, num=50)

# Create the DMPS model
DMPS_inv = DifferentialMobilityParticleSizer(DMPS_prop)

# Set up the prior. These values are given in log10-space.
expected_value = -2  # Mean of the log-normal prior
correlation_length = 12 / 16  # Controls the smoothness of the PSD estimate over the size range
log_standard_deviation = 1.0  # Controls how large variations of #/cm3 are allowed in the PSD
prior = smoothness_prior(DMPS_inv.d_m, expected_value, correlation_length, log_standard_deviation)


# Calculate the maximum a posteriori (MAP) estimate
N_MAP_W, post_cov_W = Laplace_approximation(DMPS_inv, prior, measurement)
post_std = np.sqrt(np.diag(post_cov_W))



# %% Inversion (marginalize over charger ion mobilities)

# Set up a DMPS model for inversion with marginalization over the charger ion properties
DMPS_properties_marg = DMPS_prop.copy()
DMPS_properties_marg['charging_model'] = 'LYF-interp'
DMPS_marg = DifferentialMobilityParticleSizer(DMPS_properties_marg)

posterior_samples = Laplace_approximation_marginalize(DMPS_marg, prior, measurement,
                                                      marginalize_ion_mobility=True,
                                                      marginalize_ion_ratio=False,
                                                      )



# %% Plot the results

# Figure for the results
fig, axs = plt.subplots(1, 2, num=1, clear=True)
fig.suptitle('True and estimated PSDs')

binwidth = np.log10(DMPS_inv.d_m[1]) - np.log10(DMPS_inv.d_m[0])
axs[0].fill_between(DMPS_inv.d_m * 1e9,
                    10**(N_MAP_W + 2 * post_std) / binwidth,
                    10**(N_MAP_W - 2 * post_std) / binwidth,
                    alpha=0.25, facecolor='C0', label='95 % credible interval')
plot_psd(axs[0], DMPS_inv.d_m, N=10**N_MAP_W, linestyle='--', color='k', label='MAP estimate')
plot_psd(axs[0], measurement['d_m'], n=measurement['n_true'], color='k', label='Truth')

axs[0].set_yscale('linear')
axs[0].set_xlim([6, 600])
axs[0].set_ylim([0, 23e3])
axs[0].grid('on')
axs[0].legend()
axs[0].set_title('a) MAP estimate, Wiedensohler charging model', loc='left')


plot_marginalized_psd(DMPS_marg, posterior_samples, axs[1], CI=95)

axs[1].plot(measurement['d_m'] * 1e9, measurement['n_true'], 'k-', label='Truth')

plt_max_y = 20000
plt_min_y = 0
axs[1].axis([DMPS_marg.d_m[0] * 1e9, DMPS_marg.d_m[-1] * 1e9, plt_min_y, plt_max_y])

axs[1].set_yscale('linear')
axs[1].set_xlim([6, 600])
axs[1].set_ylim([0, 23e3])
axs[1].grid('on')
axs[1].legend()
axs[1].set_title('')
axs[1].set_title('b) Marginalized posterior, LYF model', loc='left')

fig.tight_layout()

plt.show()
