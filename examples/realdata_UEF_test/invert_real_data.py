# -*- coding: utf-8 -*-

from MPSS_UQ.measurement_data import MeasurementDataset
from MPSS_UQ.particlesizers import MobilityParticleSizeSpectrometer, lpm_to_m3s
from MPSS_UQ.inversion import invert_dataset, smoothness_prior
from MPSS_UQ.analysis import (total_concentration, concentration_in_range,
                              geometric_mean_diameter, mode_diameter, median_diameter,
                              surface_area_concentration, volume_concentration, condensation_sink,
                              effective_diameter, geometric_std, relative_hdi_width)
from MPSS_UQ.plotfunctions import (plot_posterior_summary, plot_Ntot_histogram, plot_datafit,
                                   plot_timeseries_1d, plot_timeseries_2d,
                                   add_checkerboard_background)

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# =============================================================================
# Load a configuration file to set up the DMPS model
# =============================================================================

with open("../DMPS_properties.yaml", "r") as f:
    DMPS_prop = yaml.safe_load(f)
DMPS_prop = DMPS_prop['UEF-A20']


# =============================================================================
# Load the example measurement and place into MeasurementDataset
# =============================================================================

raw_data = pd.read_csv('UEF_DMPS_lvl0.txt.gz', sep='\t')

concentrations = raw_data.filter(like="conc_").to_numpy()
d_m_data = raw_data.filter(like="dmed_").iloc[0].to_numpy()

temperatures = raw_data['t_sam'].to_numpy()
pressures = raw_data['p_sam'].to_numpy()
pressures = pressures * 1e2  # Convert from hPa to Pa
datetimes = pd.to_datetime(raw_data["start_time"], utc=True).dt.tz_convert(None).to_numpy()

# Give d_m_data (output channels) to DMPS properties
DMPS_prop['d_m_data'] = d_m_data

# Convert concentration into counts
sample_flow = DMPS_prop['Qa'] * lpm_to_m3s * 1e6
counts = concentrations * sample_flow * DMPS_prop['CPC_measuring_time']

dataset = MeasurementDataset(datetimes, d_m_data, counts, 'counts', temperatures, pressures)


# =============================================================================
# Set up the inversion model
# =============================================================================

DMPS_prop['charging_model'] = 'LYF-interp'
DMPS_prop['max_charge'] = 10

DMPS = MobilityParticleSizeSpectrometer(DMPS_prop,
                                        n_bins=70,
                                        )

# Optionally, set these here if you don't plan to update them during inversion
DMPS.set_charger_properties(1.35e-4, 1.60e-4)


# =============================================================================
# Carry out inversion
# =============================================================================

# See below examples on how to select a subset of the data using datetimes
# The example dataset covers the whole of December 2024

prior = smoothness_prior(DMPS.d_m,
                         mean=0.0,
                         standard_deviation=1.5,
                         correlation_length=0.5,
                         )

psd_posteriors = invert_dataset(DMPS,
                             # dataset,
                             dataset.between_times("2024-12-10T12", "2024-12-12T12"),
                             # dataset.between_times("2024-12-08T00:00:00", "2024-12-12T00:00:00"),
                             prior=prior,
                             marginalize_ion_mobility=True,
                             parallel=True,
                             )

# You can also invert a larger dataset and then choose a subset to analyze using datetimes as
# shown below. This reuses the results objects so it is memory-light.
# example_day = psd_posteriors.between_times('2024-12-10', '2024-12-11')


# =============================================================================
# Plot the results
# =============================================================================

# How many percent of the posterior should the credible intervals cover
CI_coverage = 95

# psd_posteriors.set_reporting_range('full')
# psd_posteriors.set_reporting_range('measured')  # This is set by default

# Compute summary statistics
medians, CI_lower, CI_upper = psd_posteriors.summary(coverage=CI_coverage, n_jobs=1)


# Figure 1: Estimates and uncertainties ---------------------------------------
fig, axs = plt.subplots(nrows=4, ncols=1, num=1, clear=True)

d_m = psd_posteriors[0].d_m  # d_m of the stored results
binwidth = np.log10(d_m[1]) - np.log10(d_m[0])

# Subplot 1: Posterior medians
Z = medians.T / binwidth
plot_timeseries_2d(axs[0], psd_posteriors.datetimes, d_m, Z,
                cbar_label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$',
                vmin=1,
                cbar_extend='min',
                )
axs[0].set_title(r'Posterior median of $\mathbf{N}$')

# Subplot 2: Uncertainties as relative CI width
W = relative_hdi_width(medians, CI_lower, CI_upper).T
plot_timeseries_2d(axs[1], psd_posteriors.datetimes, d_m, W,
                cbar_label='Relative width',
                cmap='inferno_r',
                cbar_as_perc=True,
                vmax=100,
                cbar_extend='max',
                color_scale='log_3zone'
                )
axs[1].set_title(fr'Posterior uncertainty (relative width of {CI_coverage} % credible interval)')

# Subplot 3: Uncertainties posterior uncertainty reduction
R = psd_posteriors.prior_to_posterior_ratio().T
plot_timeseries_2d(axs[2], psd_posteriors.datetimes, d_m, R,
                cbar_label=r'$\sigma_\mathrm{prior} / \sigma_\mathrm{post}$',
                cmap='inferno',
                vmin=0.5
                )
axs[2].set_title('Posterior uncertainty reduction')

# Subplot 4: PSD estimate with uncertainty as the alpha channel
W_low = 1  # relative width below which estimate is ''accurate'' (alpha == 1)
W_high = 5  # relative width above which estimate is ''inaccurate'' (alpha == 0)
W_clipped = np.clip(W, W_low, W_high)
alpha = 1 - (W_clipped - W_low) / (W_high - W_low)

im2 = plot_timeseries_2d(axs[3], psd_posteriors.datetimes, d_m, Z,
                      cbar_label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$',
                      vmin=1,
                      alpha=alpha
                      )
add_checkerboard_background(axs[3], check_size_px=10, light=0.92, dark=0.75)
axs[3].set_title('Posterior median with relative uncertainty (= transparency)')

fig.tight_layout()
plt.show()



# Figure 2: Analyze single time instants in more detail -----------------------
fig = plt.figure(num=2, clear=True)

d_m = psd_posteriors[0].d_m

gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.05, 1])
axs = np.empty((2, 2), dtype=object)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
    ]

# Choose two measurements at random
idx_1, idx_2 = np.random.choice(len(psd_posteriors.datetimes), 2)

# Convert numpy datetimes to Python datetimes for easier formatting
datetime_1 = psd_posteriors.datetimes[idx_1].astype('datetime64[s]').item()
datetime_2 = psd_posteriors.datetimes[idx_2].astype('datetime64[s]').item()

plot_posterior_summary(axs[0], psd_posteriors[idx_1], CI_coverage)
axs[0].set_yscale('linear')
axs[0].set_xlim([d_m[0] * 1e9, d_m[-1] * 1e9])
axs[0].grid('on')
axs[0].set_title(
    f'Size distribution on {datetime_1.date()} at {datetime_1.time()}',
    loc='center'
    )

plot_posterior_summary(axs[2], psd_posteriors[idx_2], CI_coverage)
axs[2].set_yscale('linear')
axs[2].set_xlim([d_m[0] * 1e9, d_m[-1] * 1e9])
axs[2].grid('on')
axs[2].set_title(
    f'Size distribution on {datetime_2.date()} at {datetime_2.time()}',
    loc='center'
    )

Ntot_samples_1 = psd_posteriors[idx_1].propagate_to(total_concentration)
Ntot_samples_2 = psd_posteriors[idx_2].propagate_to(total_concentration)
plot_Ntot_histogram(axs[1], Ntot_samples_1)
plot_Ntot_histogram(axs[3], Ntot_samples_2)

line = Line2D([0.075, 0.95], [0.50, 0.50], transform=fig.transFigure,
              color='black', linewidth=4)
fig.add_artist(line)

fig.tight_layout()
plt.show()


# Figure 3: Check the datafit -------------------------------------------------
fig, ax = plt.subplots(1, 1, num=3, clear=True)

# Take the measurement matching the inverted one
meas = dataset.at_time(psd_posteriors.datetimes[idx_1])

DMPS.set_operating_conditions(meas.temperature, meas.pressure)
plot_datafit(ax, DMPS, meas.output, psd_posteriors[idx_1])



# Figure 4: Total particle numbers --------------------------------------------
fig, axs = plt.subplots(nrows=4, ncols=1, num=4, clear=True)

d_m = psd_posteriors[0].d_m

Ntot_samples = psd_posteriors.propagate_to(total_concentration)
nucl_samples = psd_posteriors.propagate_to(concentration_in_range, d_m, 1e-9, 25e-9)
ait_samples = psd_posteriors.propagate_to(concentration_in_range, d_m, 25e-9, 100e-9)
acc_samples = psd_posteriors.propagate_to(concentration_in_range, d_m, 100e-9, 1000e-9)

dm0 = 10**(np.log10(psd_posteriors[0].d_m[0]) - 0.5 * binwidth) * 1e9
dm1 = 10**(np.log10(psd_posteriors[0].d_m[-1]) + 0.5 * binwidth) * 1e9

plot_timeseries_1d(axs[0], psd_posteriors.datetimes, Ntot_samples, coverage=CI_coverage,
                   title=rf'$N_\mathrm{{tot}}$ between [{dm0 : .1f}, {dm1 : .1f}] nm')
plot_timeseries_1d(axs[1], psd_posteriors.datetimes, nucl_samples, coverage=CI_coverage,
                   title=r'$N_\mathrm{tot}$ in nucleation mode (1-25 nm)')
plot_timeseries_1d(axs[2], psd_posteriors.datetimes, ait_samples, coverage=CI_coverage,
                   title=r'$N_\mathrm{tot}$ in Aitken mode (25-100 nm)')
plot_timeseries_1d(axs[3], psd_posteriors.datetimes, acc_samples, coverage=CI_coverage,
                   title=r'$N_\mathrm{tot}$ in accumulation mode (100-1000 nm)')
fig.tight_layout()



# Figures 5-6: Plot some derived quantities -----------------------------------
gmd_samples = psd_posteriors.propagate_to(geometric_mean_diameter, d_m)
gstd_samples = psd_posteriors.propagate_to(geometric_std, d_m)
mode_samples = psd_posteriors.propagate_to(mode_diameter, d_m)
median_samples = psd_posteriors.propagate_to(median_diameter, d_m)
surf_samples = psd_posteriors.propagate_to(surface_area_concentration, d_m)
vol_samples = psd_posteriors.propagate_to(volume_concentration, d_m)
CS_samples = psd_posteriors.propagate_to(condensation_sink, d_m)
eff_diam_samples = psd_posteriors.propagate_to(effective_diameter, d_m)


fig, axs = plt.subplots(4, 1, num=5, clear=True)
fig.suptitle('Growth and surface area effects')

plot_timeseries_1d(axs[0], psd_posteriors.datetimes, CS_samples, coverage=CI_coverage,
                   title=r'Condensation sink')
axs[0].set_ylabel(r's$^{-1}$')
plot_timeseries_1d(axs[1], psd_posteriors.datetimes, surf_samples, coverage=CI_coverage,
                   title=r'Total surace area concentration')
axs[1].set_ylabel(r'$\mu$m$^2$ cm$^{-3}$')
plot_timeseries_1d(axs[2], psd_posteriors.datetimes, vol_samples, coverage=CI_coverage,
                   title=r'Total volume concentration')
axs[2].set_ylabel(r'$\mu$m$^3$ cm$^{-3}$')
plot_timeseries_1d(axs[3], psd_posteriors.datetimes, eff_diam_samples, coverage=CI_coverage,
                   title=r'Effective diameter')
axs[3].set_ylabel(r'$d_m$')
fig.tight_layout()


fig, axs = plt.subplots(4, 1, num=6, clear=True)
fig.suptitle('Shape evolution')

ymin = np.floor(dataset[0].d_m_data[0] * 1e8) * 1e-8
ymax = np.ceil(dataset[0].d_m_data[-1] * 1e8) * 1e-8 * 0.5
plot_timeseries_1d(axs[0], psd_posteriors.datetimes, gmd_samples, coverage=CI_coverage,
                   title='Geometric mean diameter', log_yscale=True, ymin=ymin, ymax=ymax)
axs[0].set_ylabel(r'$d_m$')
plot_timeseries_1d(axs[1], psd_posteriors.datetimes, gstd_samples, coverage=CI_coverage,
                   title='Geometric std', ymin=1)
plot_timeseries_1d(axs[2], psd_posteriors.datetimes, mode_samples, coverage=CI_coverage,
                   title=r'Mode diameter', log_yscale=True, ymin=ymin, ymax=ymax)
axs[2].set_ylabel(r'$d_m$')
plot_timeseries_1d(axs[3], psd_posteriors.datetimes, median_samples, coverage=CI_coverage,
                   title=r'Median diameter', log_yscale=True, ymin=ymin, ymax=ymax)
axs[3].set_ylabel(r'$d_m$')
fig.tight_layout()
