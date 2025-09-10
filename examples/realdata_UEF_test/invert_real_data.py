# -*- coding: utf-8 -*-

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as tck
import zipfile
import os

from tqdm import tqdm

from MPSS_UQ.measurement_data import MeasurementDataset, measurement_loader
from MPSS_UQ.particlesizers import DifferentialMobilityParticleSizer, lpm_to_m3s
from MPSS_UQ.inversion import compute_posterior, smoothness_prior

from read_dmps_files_labtest import load_and_process_data


# =============================================================================
# Load a configuration file to set up the DMPS model
# =============================================================================

with open("DMPS_properties.yaml", "r") as f:
    DMPS_prop = yaml.safe_load(f)
DMPS_prop = DMPS_prop['UEF-A20']



# =============================================================================
# Load the example measurement and place into MeasurementDataset
# =============================================================================

folder_path = "UEF_DMPS_level_0_test_data"

# Extract the .zip file if needed
if not os.path.exists(folder_path):
    with zipfile.ZipFile("UEF_DMPS_level_0_test_data.zip", 'r') as zip_ref:
        zip_ref.extractall("UEF_DMPS_level_0_test_data")

start_date = "2024-11-13"
end_date = "2024-11-20"
filename = 'UEF_DMPS_level_0_'

df_lvl0_dmps = load_and_process_data(filename, folder_path, start_date, end_date)

conc_columns = [col for col in df_lvl0_dmps.columns if col.startswith('conc')]
d_m_columns = [col for col in df_lvl0_dmps.columns if col.startswith('dm')]

concentrations = df_lvl0_dmps[conc_columns].to_numpy()
d_m_data = df_lvl0_dmps[d_m_columns].iloc[0].to_numpy()

# Give d_m_data (output channels) to DMPS properties
DMPS_prop['d_m_data'] = d_m_data

temperatures = df_lvl0_dmps['t_sam'].to_numpy()
pressures = df_lvl0_dmps['p_sam'].to_numpy()
datetimes = df_lvl0_dmps.index.values

# Convert concentration into counts
sample_flow = DMPS_prop['Qa'] * lpm_to_m3s * 1e6
counts = concentrations * sample_flow * DMPS_prop['CPC_measuring_time']

dataset = MeasurementDataset(datetimes, d_m_data, counts, 'counts', temperatures, pressures)



# =============================================================================
# Set up the inversion model
# =============================================================================

# Mobility diameters for the inverted PSD
DMPS_prop['d_m'] = np.geomspace(6e-9, 1000e-9, num=50)

DMPS_prop['charging_model'] = 'LYF-interp'
DMPS_prop['max_charge'] = 4

DMPS_inv = DifferentialMobilityParticleSizer(DMPS_prop)
DMPS_inv.set_charger_properties(1.35e-4, 1.60e-4, 1)
DMPS_inv.set_operating_conditions(290, 101325)

# Preallocate
N_MAP = np.zeros((len(dataset), DMPS_prop['d_m'].shape[0]))
CI_lower = np.zeros_like(N_MAP)
CI_upper = np.zeros_like(N_MAP)

# Configure the prior
expected_value = -2
correlation_length = 8 / 16
log_standard_deviation = 2.0
prior = smoothness_prior(DMPS_prop['d_m'], expected_value,
                         correlation_length, log_standard_deviation
                         )



# =============================================================================
# Carry out inversion
# =============================================================================

CI_percent = 95
for idx, measurement in enumerate(tqdm(measurement_loader(dataset), total=len(dataset))):
    DMPS_inv.set_operating_conditions(measurement.temperature,
                                      measurement.pressure * 1e2
                                      )
    N_MAP[idx], CI_lower[idx], CI_upper[idx] = compute_posterior(DMPS_inv,
                                                                 prior,
                                                                 measurement,
                                                                 CI=CI_percent
                                                                 )



# =============================================================================
# Plot the results
# =============================================================================

fig, ax0 = plt.subplots(num=10, clear=True)

binwidth = np.log10(DMPS_inv.d_m[1]) - np.log10(DMPS_inv.d_m[0])
Z = N_MAP.T / binwidth

plt_N_min = 10**0
plt_N_max = np.max(Z)
im = ax0.pcolormesh(*np.meshgrid(datetimes, DMPS_inv.d_m * 1e9), Z,
                   norm=colors.LogNorm(vmin=plt_N_min, vmax=plt_N_max),
                   cmap='viridis')

ax0.set_yscale('log')
ax0.yaxis.set_major_formatter(tck.FormatStrFormatter('%.0f'))
ax0.set_yticks([DMPS_inv.d_m[0] * 1e9, 10, 20, 50, 100, 250, 500, DMPS_inv.d_m[-1] * 1e9])

ax0.set_ylabel('Particle diameter (nm)')
ax0.set_xlabel('Time')
ax0.set_title('Inverted particle size distribution of a DMPS measurement')
            
cbar = fig.colorbar(im, ax=ax0,
                     label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$')

fig.tight_layout()
