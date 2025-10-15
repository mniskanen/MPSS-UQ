# -*- coding: utf-8 -*-

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from MPSS_UQ.particlesizers import DifferentialMobilityParticleSizer
from MPSS_UQ.inversion import invert_psd, smoothness_prior
from MPSS_UQ.measurement_data import generate_DMPS_measurement
from MPSS_UQ.plotfunctions import plot_psd, plot_posterior_summary, plot_Ntot_histogram


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

# Choose mobility diameters the DMPS measures (i.e. the output channels)
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

# Choose the mobility diameters for the inverted PSD
DMPS_prop['d_m'] = np.geomspace(6e-9, 1000e-9, num=50)


# Set up the Gaussian smoothness prior. The values are given in log10-space.
# The values needed to fully specify the prior are the expected (i.e. mean) value,
# the correlation length which controls the smoothness of the result w.r.t. particle size,
# and the standard deviation which controls how large variations in the concentration
# are allowed.
expected_value = -2
correlation_length = 12 / 16
log_standard_deviation = 1.5
prior = smoothness_prior(DMPS_prop['d_m'],
                         expected_value,
                         correlation_length,
                         log_standard_deviation
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

result = invert_psd(DMPS_inv, measurement, prior)



# =============================================================================
# Step 4: Carry out inversion - Option 2: Marginalize over ion properties
# =============================================================================

DMPS_properties_marg = DMPS_prop.copy()

# To marginalize, we have to use a model where we can modify the ion properties,
# for example the LYF-interp model
DMPS_properties_marg['charging_model'] = 'LYF-interp'

DMPS_marg = DifferentialMobilityParticleSizer(DMPS_properties_marg)

DMPS_marg.set_operating_conditions(measurement.temperature,
                                   measurement.pressure
                                   )

# Marginalize over charger ion mobilities
result_marg = invert_psd(DMPS_marg,
                         measurement,
                         prior,
                         marginalize_ion_mobility=True,
                         num_samples=100000,  # more samples for cleaner plots
                         )



# =============================================================================
# Step 5: Plot the results
# =============================================================================

CI_coverage = 95

# To return the estimate mean value and lower and upper limits of the credible intervals, do:
# mean, CI_lower, CI_upper = result.posterior_summary(coverage=CI_coverage)

# fig, axs = plt.subplots(2, 2, num=1, clear=True)
fig = plt.figure(num=1, clear=True)
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.05, 1])  # middle row is a gap for a 
axs = np.empty((2, 2), dtype=object)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
    ]

fig.suptitle('True and estimated PSDs')

plot_posterior_summary(axs[0], result, CI_coverage)
plot_psd(axs[0], measurement.d_m_truth, measurement.N_true, color='k', label='Truth')
axs[0].set_yscale('linear')
axs[0].set_xlim([DMPS_marg.d_m[0] * 1e9, DMPS_marg.d_m[-1] * 1e9])
axs[0].grid('on')
axs[0].legend()
axs[0].set_title('a) PSD estimate, charging uncertainty not considered', loc='left')


plot_posterior_summary(axs[2], result_marg, CI_coverage)
plot_psd(axs[2], measurement.d_m_truth, measurement.N_true, color='k', label='Truth')
axs[2].set_yscale('linear')
axs[2].set_xlim([DMPS_marg.d_m[0] * 1e9, DMPS_marg.d_m[-1] * 1e9])
axs[2].grid('on')
axs[2].legend()
axs[2].set_title('b) PSD estimate, marginalized charging uncertainty', loc='left')

# Set the same ylimits for both graphs
y0_min, y0_max = axs[0].get_ylim()
y1_min, y1_max = axs[2].get_ylim()
y_min = np.min((y0_min, y1_min))
y_max = np.max((y0_max, y1_max))

axs[0].set_ylim([y_min, y_max])
axs[2].set_ylim([y_min, y_max])


# Calculate the true Ntot only for the sizes we consider in the inversion
true_idx_1 = np.where(measurement.d_m_truth >= DMPS_prop['d_m'][0])[0][0]
true_idx_2 = np.where(measurement.d_m_truth <= DMPS_prop['d_m'][-1])[0][-1]
Ntot_true = np.sum(measurement.N_true[true_idx_1 : true_idx_2 + 1])
Ntot_samples = result.Ntot_samples()
Ntot_samples_marg = result_marg.Ntot_samples()
xlimits = (min(np.min(Ntot_samples), np.min(Ntot_samples_marg)),
           max(np.max(Ntot_samples), np.max(Ntot_samples_marg)),
           )
plot_Ntot_histogram(axs[1], Ntot_samples, Ntot_true=Ntot_true, xlimits=xlimits)
plot_Ntot_histogram(axs[3], Ntot_samples_marg, Ntot_true=Ntot_true, xlimits=xlimits)


line = Line2D([0.075, 0.95], [0.49, 0.49], transform=fig.transFigure,
              color='black', linewidth=4)
fig.add_artist(line)

fig.tight_layout()
plt.show()
