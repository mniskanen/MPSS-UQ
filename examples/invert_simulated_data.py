# -*- coding: utf-8 -*-

import yaml
import numpy as np
import matplotlib.pyplot as plt

from MPSS_UQ.particlesizers import DifferentialMobilityParticleSizer
from MPSS_UQ.inversion import (compute_posterior,
                               compute_posterior_marginalize,
                               Laplace_approximation_marginalize,
                               smoothness_prior
                               )
from MPSS_UQ.measurement_data import generate_DMPS_measurement
from MPSS_UQ.plotfunctions import plot_psd, plot_marginalized_psd, plot_posterior_summary


'''
This script shows an example on how to use this package for inversion of DMPS data.
In this example we use synthetic data generated with a DMPS model.
'''

# The workflow consists of the following steps:


# =============================================================================
# Step 1: Load a configuration file to set up the DMPS model
# =============================================================================

# Load a DMPS configuration file. This includes basic geometry and flow rates information
# on the DMPS. The file DMPS_properties.yaml should be in the same folder as this script.
with open("DMPS_properties.yaml", "r") as f:
    DMPS_prop = yaml.safe_load(f)

# Choose the DMPS
DMPS_prop = DMPS_prop['UEF-A20']

# Mobility diameters the DMPS measures (i.e. the output channels)
DMPS_prop['d_m_data'] = np.geomspace(6e-9, 800e-9, num=30) # d_min, d_max, num_bins


# To use a custom form for the CPC counting efficiency, specify it here and add it to the
# DMPS_prop dictionary under a key 'custom_CPC_count_eff_function'. The function should take
# as input the mobility diameters in [m] and return the counting efficiency curve.

# def custom_CPC_counting_efficiency(d_m):
#     count_eff = 1.00 * (1 - np.exp(np.log(2) * ((7.5 - d_m * 1e9) / (10.0 - 7.5))))
#     return np.clip(count_eff, 0, 1)

# DMPS_prop['custom_CPC_count_eff_function'] = custom_CPC_counting_efficiency

    

# =============================================================================
# Step 2: Generate synthetic data
# =============================================================================

# Predefined PSD scenarios are:
#     Urban, Marine, Rural, Remote continental, Free troposphere, Polar, Desert
measurement = generate_DMPS_measurement(DMPS_prop.copy(), scenario='Urban')



# =============================================================================
# Step 3: Set up the inversion model
# =============================================================================

# Mobility diameters for the inverted PSD
DMPS_prop['d_m'] = np.geomspace(6e-9, 1000e-9, num=50)


# Set up the smoothness prior. The values are given in log10-space.

# Mean of the log-normal prior
expected_value = -2

# Controls the smoothness of the PSD estimate over the size range
correlation_length = 12 / 16

# Controls how large variations of #/cm3 are allowed in the PSD
log_standard_deviation = 1.5

prior = smoothness_prior(DMPS_prop['d_m'], expected_value,
                         correlation_length, log_standard_deviation
                         )



# =============================================================================
# Step 4: Carry out inversion - Option 1: Laplace approximation
# =============================================================================

# Choose a charging model
DMPS_prop['charging_model'] = 'Wiedensohler'
DMPS_prop['max_charge'] = 4

# Create the DMPS object used in the inversion
DMPS_inv = DifferentialMobilityParticleSizer(DMPS_prop)

DMPS_inv.set_operating_conditions(measurement.temperature,
                                  measurement.pressure
                                  )

result = compute_posterior(DMPS_inv, prior, measurement)



# =============================================================================
# Step 4: Carry out inversion - Option 2: Marginalize over ion properties
# =============================================================================

DMPS_properties_marg = DMPS_prop.copy()

# To marginalize, we have to use a model where we can modify the ion properties,
# for example the LYF-interp model
DMPS_properties_marg['charging_model'] = 'LYF-interp-fast'

DMPS_marg = DifferentialMobilityParticleSizer(DMPS_properties_marg)

DMPS_marg.set_operating_conditions(measurement.temperature,
                                   measurement.pressure
                                   )

# Marginalize over charger ion mobilities
result_marg = compute_posterior_marginalize(DMPS_marg,
                                            prior,
                                            measurement,
                                            )



# =============================================================================
# Step 5: Plot the results
# =============================================================================

CI_coverage = 95

fig, axs = plt.subplots(1, 2, num=1, clear=True)
fig.suptitle('True and estimated PSDs')

plot_posterior_summary(axs[0], result, CI_coverage)
plot_psd(axs[0], measurement.d_m_truth, measurement.N_true, color='k', label='Truth')
axs[0].set_yscale('linear')
axs[0].set_xlim([DMPS_marg.d_m[0] * 1e9, DMPS_marg.d_m[-1] * 1e9])
axs[0].grid('on')
axs[0].legend()
axs[0].set_title('a) MAP estimate, Wiedensohler charging model', loc='left')


plot_posterior_summary(axs[1], result_marg, CI_coverage)
plot_psd(axs[1], measurement.d_m_truth, measurement.N_true, color='k', label='Truth')
axs[1].set_yscale('linear')
axs[1].set_xlim([DMPS_marg.d_m[0] * 1e9, DMPS_marg.d_m[-1] * 1e9])
axs[1].grid('on')
axs[1].legend()
axs[1].set_title('b) Marginalized posterior, LYF model', loc='left')

# Set the same ylimits for both graphs
_, y0_max = axs[0].get_ylim()
_, y1_max = axs[1].get_ylim()
y_max = np.max((y0_max, y1_max))

axs[0].set_ylim([0, y_max])
axs[1].set_ylim([0, y_max])

fig.tight_layout()

plt.show()
