# -*- coding: utf-8 -*-

import yaml
import numpy as np
import matplotlib.pyplot as plt

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

# The workflow consists of the following steps:

    
# =============================================================================
# Step 1: Load configuration to set up the DMPS model
# =============================================================================

# Load a DMPS configuration file. This includes basic geometry and flow rates information
# on the DMPS. The file DMPS_properties.json should be in the same folder as this script.
with open("DMPS_properties.yaml", "r") as f:
    DMPS_prop = yaml.safe_load(f)

# Choose the DMPS
DMPS_prop = DMPS_prop['UEF-A20']


# To use a custom form for the CPC counting efficiency, specify it here and add it to the
# DMPS_prop dictionary under a key 'custom_CPC_count_eff_function'. The function should take
# as input the mobility diameters in [m] and return the counting efficiency curve.

# def custom_CPC_counting_efficiency(d_m):
#     count_eff = 1.00 * (1 - np.exp(np.log(2) * ((7.5 - d_m * 1e9) / (10.0 - 7.5))))
#     return np.clip(count_eff, 0, 1)

# DMPS_prop['custom_CPC_count_eff_function'] = custom_CPC_counting_efficiency

    

# =============================================================================
# Step 2, option 1: Generate synthetic data
# =============================================================================

# Set up a separate DMPS for generating the data
DMPS_prop_datagen = DMPS_prop.copy()

# Mobility diameters that are used to represent the true PSD
DMPS_prop_datagen['d_m'] = np.geomspace(2e-9, 2500e-9, num=500)  # d_min, d_max, num_bins

# Mobility diameters the DMPS measures (i.e. the output channels)
DMPS_prop_datagen['d_m_data'] = np.geomspace(6e-9, 500e-9, num=30)

# Which bipolar charging model to use
DMPS_prop_datagen['charging_model'] = 'LYF-interp'

# Maximum considered number of multiple charges
DMPS_prop_datagen['max_charge'] = 8

# Create the data generating DMPS object
DMPS_datagen = DifferentialMobilityParticleSizer(DMPS_prop_datagen)

# Set charger ion properties for the measurement (if using the LYF model)
pos_ion_mobility = 1.4e-4
neg_ion_mobility = 1.9e-4
ion_ratio = 1.0
DMPS_datagen.update_charger_ion_properties(pos_ion_mobility, neg_ion_mobility, ion_ratio)

# Create the measurement
measurement = generate_DMPS_measurement(DMPS_datagen, scenario='Urban')


# =============================================================================
# Step 2, option 2: Load measurement data
# =============================================================================



# =============================================================================
# Step 3: Set up the inversion model
# =============================================================================

# Mobility diameters the DMPS measures
DMPS_prop['d_m_data'] = np.geomspace(6e-9, 500e-9, num=30)

# Mobility diameters for the inverted PSD
DMPS_prop['d_m'] = np.geomspace(6e-9, 2500e-9, num=50)

# Set up the prior. These values are given in log10-space.
# Currently the only option is to use the smoothness prior.

# Mean of the log-normal prior
expected_value = -2

# Controls the smoothness of the PSD estimate over the size range
correlation_length = 12 / 16

# Controls how large variations of #/cm3 are allowed in the PSD
log_standard_deviation = 1.0

prior = smoothness_prior(DMPS_prop['d_m'], expected_value,
                         correlation_length, log_standard_deviation
                         )



# =============================================================================
# Step 4: Carry out inversion - Option 1: Laplace approximation
# =============================================================================

# Charging model set up
DMPS_prop['charging_model'] = 'Wiedensohler'
DMPS_prop['max_charge'] = 4

# Create the DMPS object used in the inversion
DMPS_inv = DifferentialMobilityParticleSizer(DMPS_prop)

# Laplace approximation
N_MAP_W, post_cov_W = Laplace_approximation(DMPS_inv, prior, measurement)


# =============================================================================
# Step 4: Carry out inversion - Option 2: Marginalize over ion properties
# =============================================================================

# Have to use the LYF-interp model here
DMPS_properties_marg = DMPS_prop.copy()
DMPS_properties_marg['charging_model'] = 'LYF-interp'

DMPS_marg = DifferentialMobilityParticleSizer(DMPS_properties_marg)

# Marginalize over charger ion mobilities
posterior_samples = Laplace_approximation_marginalize(DMPS_marg, prior, measurement,
                                                      marginalize_ion_mobility=True,
                                                      marginalize_ion_ratio=False,
                                                      )



# =============================================================================
# Step 5: Plot the results
# =============================================================================

fig, axs = plt.subplots(1, 2, num=1, clear=True)
fig.suptitle('True and estimated PSDs')

binwidth = np.log10(DMPS_inv.d_m[1]) - np.log10(DMPS_inv.d_m[0])
post_std = np.sqrt(np.diag(post_cov_W))
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
