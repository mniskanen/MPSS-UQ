# -*- coding: utf-8 -*-

from MPSS_UQ.measurement_data import MeasurementDataset
from MPSS_UQ.particlesizers import DifferentialMobilityParticleSizer, lpm_to_m3s
from MPSS_UQ.inversion import invert_dataset, smoothness_prior
from MPSS_UQ.plotfunctions import (plot_posterior_summary, plot_Ntot_histogram, plot_datafit,
                                   plot_timeseries)

import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

# # select only some measurements
# raw_data = raw_data.iloc[3000:3500]

concentrations = raw_data.filter(like="conc_").to_numpy()
d_m_data = raw_data.filter(like="dmed_").iloc[0].to_numpy()

temperatures = raw_data['t_sam'].to_numpy()
pressures = raw_data['p_sam'].to_numpy()
pressures *= 1e2  # Convert from hPa to Pa
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

DMPS = DifferentialMobilityParticleSizer(DMPS_prop, n_bins=70)

# Optionally, set these here if you don't plan to update them during inversion
DMPS.set_charger_properties(1.35e-4, 1.60e-4)



# =============================================================================
# Carry out inversion
# =============================================================================

# See below examples on how to select a subset of the data using datetimes
# The example dataset covers the whole of December 2024

inv_dataset = invert_dataset(DMPS,
                             # dataset,
                             dataset.between_times("2024-12-09", "2024-12-12"),
                             # dataset.between_times("2024-12-01T00:00:00", "2024-12-03T12:00:00"),
                             marginalize_ion_mobility=True,
                             parallel=True,
                             )

# You can also invert a larger dataset and then choose a subset to analyze using datetimes as
# shown below. This reuses the results objects so it is memory-light.
# example_day = inv_dataset.between_times('2024-12-10', '2024-12-11')



# =============================================================================
# Plot the results
# =============================================================================

# How many percent of the posterior should the credible intervals cover
CI_coverage = 95

# inv_dataset.set_reporting_range('full')
# inv_dataset.set_reporting_range('measured')  # This is set by default



# Figure 1: Estimates and uncertainties ---------------------------------------
fig, axs = plt.subplots(nrows=3, ncols=1, num=1, clear=True)

# Compute summary statistics
medians, CI_lower, CI_upper = inv_dataset.posterior_summary(coverage=CI_coverage)

d_m = inv_dataset.results[0].d_m  # d_m of the stored results
binwidth = np.log10(d_m[1]) - np.log10(d_m[0])

# Subplot 1: Posterior medians
Z = medians.T / binwidth
plot_timeseries(axs[0], inv_dataset.datetimes, d_m, Z,
                cbar_label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$',
                )
axs[0].set_title(r'Posterior median of $\mathbf{N}$')

# Subplot 2: Uncertainties as relative CI width
# Safeguard for zero/near-zero lower bounds
eps = 0.0001
W = ((CI_upper - CI_lower) / (medians + eps)).T

plot_timeseries(axs[1], inv_dataset.datetimes, d_m, W,
                # log_color_scale=False,
                cbar_label=fr'Relative {CI_coverage} % HDI width',
                cmap='inferno',
                # vmin=0.2,
                # vmax=140,
                )
axs[1].set_title(fr'Uncertainty (relative {CI_coverage} % HDI width)')


# Subplot 3: PSD estimate with uncertainty as the alpha channel
norm_psd = colors.LogNorm(vmin=np.nanquantile(Z, 0.001),
                           vmax=np.nanmax(Z))
cmap_psd = mpl.colormaps['viridis']

# Choose values for W
W_low = 1  # width below which estimate is ''accurate'' (alpha == 1)
W_high = 10  # width above which estimate is ''inaccurate'' (alpha == 0)
W_clipped = np.clip(W, W_low, W_high)
alpha = 1 - (W_clipped - W_low) / (W_high - W_low)
rgba = cmap_psd(norm_psd(Z))
rgba[..., -1] = alpha

# Draw empty mesh, then set RGBA facecolors
im2 = plot_timeseries(axs[2], inv_dataset.datetimes, d_m, Z,
                      cbar_label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$',
                      )
im2.set_cmap(None)
im2.set_norm(None)
im2.set_array(None)
im2.set_facecolors(rgba.reshape(-1, 4))

# Colorbar from PSD only (not alpha)
sm = plt.cm.ScalarMappable(norm=norm_psd, cmap=cmap_psd)
sm.set_array([])
axs[2]._my_colorbar.update_normal(sm)
axs[2].set_title('Posterior median with relative uncertainty (= transparency)')

fig.tight_layout()
plt.show()



# Figure 2: Variance reduction and posterior CI width plots -------------------
fig, axs = plt.subplots(nrows=2, ncols=1, num=2, clear=True)

prior = smoothness_prior(d_m, 0, 8/16, 1.5)
prior_variance = np.diag(prior['covariance'])
post_variance = inv_dataset.posterior_variance()
VR = np.log10(post_variance / prior_variance).T

plot_timeseries(axs[0], inv_dataset.datetimes, d_m, VR,
                log_color_scale=False,
                cbar_label=r'$\log_{10}(\sigma_\mathrm{post}^2 / \sigma_\mathrm{prior}^2)$',
                cmap='Blues_r'
                )
axs[0].set_title('log variance reduction')

# Plot uncertainties as CI width
CI_width = CI_upper - CI_lower
plotval = CI_width.T / binwidth
vmin = np.quantile(plotval.flatten(), 0.001)
plot_timeseries(axs[1], inv_dataset.datetimes, d_m, plotval,
                # log_color_scale=False,
                cbar_label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$',
                cmap='Blues_r'
                )
axs[1].set_title(f'Estimate uncertainty (width of the {CI_coverage} % credible intervals)')

fig.tight_layout()
plt.show()



# Figure 3: Analyze single time instants in more detail -----------------------
fig = plt.figure(num=3, clear=True)

gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.05, 1])
axs = np.empty((2, 2), dtype=object)
axs = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[2, 0]),
    fig.add_subplot(gs[2, 1]),
    ]

# Choose two measurements at random
idx_1, idx_2 = np.random.choice(len(inv_dataset.datetimes), 2)

# Convert numpy datetimes to Python datetimes for easier formatting
datetime_1 = inv_dataset.datetimes[idx_1].astype('datetime64[s]').item()
datetime_2 = inv_dataset.datetimes[idx_2].astype('datetime64[s]').item()

plot_posterior_summary(axs[0], inv_dataset.results[idx_1], CI_coverage)
axs[0].set_yscale('linear')
axs[0].set_xlim([d_m[0] * 1e9, d_m[-1] * 1e9])
axs[0].grid('on')
axs[0].set_title(
    f'Size distribution on {datetime_1.date()} at {datetime_1.time()}',
    loc='center'
    )

plot_posterior_summary(axs[2], inv_dataset.results[idx_2], CI_coverage)
axs[2].set_yscale('linear')
axs[2].set_xlim([d_m[0] * 1e9, d_m[-1] * 1e9])
axs[2].grid('on')
axs[2].set_title(
    f'Size distribution on {datetime_2.date()} at {datetime_2.time()}',
    loc='center'
    )

Ntot_samples_1 = inv_dataset.results[idx_1].Ntot_samples()
Ntot_samples_2 = inv_dataset.results[idx_2].Ntot_samples()
plot_Ntot_histogram(axs[1], Ntot_samples_1)
plot_Ntot_histogram(axs[3], Ntot_samples_2)

line = Line2D([0.075, 0.95], [0.50, 0.50], transform=fig.transFigure,
              color='black', linewidth=4)
fig.add_artist(line)

fig.tight_layout()
plt.show()


# Figure 4: Total particle numbers --------------------------------------------
fig, ax = plt.subplots(nrows=1, ncols=1, num=4, clear=True)

Ntots, Ntots_CI_low, Ntots_CI_high = inv_dataset.Ntot_summary(coverage=95)

ax.fill_between(inv_dataset.datetimes,
                Ntots_CI_low,
                Ntots_CI_high,
                alpha=0.25,
                facecolor='C0',
                label=f'{CI_coverage} % credible interval'
                )
ax.plot(inv_dataset.datetimes, Ntots, label=r'Median $N_\mathrm{tot}$')
ax.set_xlabel('Time')
ax.set_ylabel('Ntot')
dm0 = 10**(np.log10(inv_dataset.results[0].d_m[0]) - 0.5 * binwidth) * 1e9
dm1 = 10**(np.log10(inv_dataset.results[0].d_m[-1]) + 0.5 * binwidth) * 1e9
ax.set_title(f'Total particle numbers between [{dm0 : .1f}, {dm1 : .1f}] nm')
ax.grid('on')
ax.legend()



# Figure 5: Check the datafit -------------------------------------------------
fig, ax = plt.subplots(1, 1, num=5, clear=True)

DMPS.set_operating_conditions(dataset[idx_1].temperature, dataset[idx_1].pressure)
plot_datafit(ax, DMPS, dataset[idx_1].output, inv_dataset.results[idx_1])
