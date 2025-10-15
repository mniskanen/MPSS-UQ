# -*- coding: utf-8 -*-

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as tck
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import psutil
from joblib import Parallel, delayed
import zipfile
import os

from tqdm import tqdm

from MPSS_UQ.measurement_data import MeasurementDataset, measurement_loader
from MPSS_UQ.particlesizers import DifferentialMobilityParticleSizer, lpm_to_m3s
from MPSS_UQ.inversion import invert_psd, smoothness_prior
from MPSS_UQ.inversion_results import InversionDataset
from MPSS_UQ.plotfunctions import plot_posterior_summary, plot_Ntot_histogram

from read_dmps_files_labtest import load_and_process_data



DO_PARALLEL = False
MARGINALIZE_ION_MOBILITY = False


if __name__ == '__main__':

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
    DMPS_prop['d_m'] = np.geomspace(5e-9, 2500e-9, num=60)
    
    # DMPS_prop['charging_model'] = 'LYF-interp-flux'
    DMPS_prop['charging_model'] = 'LYF-interp'
    # DMPS_prop['charging_model'] = 'Wiedensohler'
    DMPS_prop['max_charge'] = 4
    
    DMPS = DifferentialMobilityParticleSizer(DMPS_prop)
    DMPS.set_charger_properties(1.35e-4, 1.60e-4)#, 1)
    DMPS.set_operating_conditions(290, 101325)
    
    # Configure the prior
    expected_value = -2
    correlation_length = 5 / 16
    log_standard_deviation = 1.5
    prior = smoothness_prior(DMPS_prop['d_m'], expected_value,
                             correlation_length, log_standard_deviation
                             )
    
    # Storage for the inversion results
    inv_dataset = InversionDataset(datetimes)
    
    
    # =============================================================================
    # Carry out inversion
    # =============================================================================
    
    if DO_PARALLEL:
        
        # Wrapper functions to get the iteration number for parallel execution
        def run_inversion(args):
            idx, measurement = args
            result = invert_psd(DMPS, measurement, prior,
                                marginalize_ion_mobility=MARGINALIZE_ION_MOBILITY)
            return idx, result
        
        # Set the number of processes
        n_cpus = psutil.cpu_count(logical=False)
        n_jobs = n_cpus - 1  # Leave at least one thread free for other use
        
        # Collect all measurements into a list
        all_measurements = [dataset[i] for i in range(len(dataset))]

        # Prepare all measurements for parallel execution
        args = list(enumerate(all_measurements))
        
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(run_inversion)(args) for args in tqdm(args)
            )
        
        # Store results
        for idx, result in results:
            inv_dataset.assign_result(idx, result)
    
    else:
        
        for idx, measurement in enumerate(tqdm(measurement_loader(dataset), total=len(dataset))):
            # DMPS.set_operating_conditions(measurement.temperature,
            #                                   measurement.pressure * 1e2
            #                                   )
            
            result = invert_psd(DMPS, measurement, prior,
                                marginalize_ion_mobility=MARGINALIZE_ION_MOBILITY)
            
            inv_dataset.assign_result(idx, result)
    
    
    
    #%%
    # =============================================================================
    # Plot the results
    # =============================================================================
    
    # How many percent of the posterior should the credible intervals cover
    CI_coverage = 95
    
    # Summarize the posterior of each measurement
    means, CI_lower, CI_upper = inv_dataset.posterior_summary(coverage=CI_coverage)
    
    fig, axs = plt.subplots(nrows=2, ncols=1, num=10, clear=True)
    binwidth = np.log10(DMPS.d_m[1]) - np.log10(DMPS.d_m[0])
    
    Z = means.T / binwidth
    
    plt_N_min = 10**0#np.min(Z)
    plt_N_max = np.max(Z)
    im = axs[0].pcolormesh(*np.meshgrid(datetimes, DMPS.d_m * 1e9), Z,
                       norm=colors.LogNorm(vmin=plt_N_min, vmax=plt_N_max),
                       cmap='viridis')
    
    axs[0].set_yscale('log')
    axs[0].yaxis.set_major_formatter(tck.FormatStrFormatter('%.0f'))
    axs[0].set_yticks([DMPS.d_m[0] * 1e9, 10, 20, 50, 100, 250, 500, DMPS.d_m[-1] * 1e9])
    
    axs[0].set_ylabel('Particle diameter (nm)')
    axs[0].set_xlabel('Time')
    axs[0].set_title('Inverted particle size distribution of a DMPS measurement')
                
    cbar = fig.colorbar(im, ax=axs[0],
                         label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$')
    
    # Plot uncertainties
    CI_width = CI_upper - CI_lower
    plotval = CI_width.T / binwidth
    plt_CIw_min = np.min(plotval)  #10**-1.5
    plt_CIw_max = np.max(plotval)
    im = axs[1].pcolormesh(*np.meshgrid(datetimes, DMPS.d_m * 1e9), plotval,
                       norm=colors.LogNorm(vmin=plt_CIw_min, vmax=plt_CIw_max),
                       cmap='Blues_r')
    
    axs[1].set_yscale('log')
    axs[1].yaxis.set_major_formatter(tck.FormatStrFormatter('%.0f'))
    axs[1].set_yticks([DMPS.d_m[0] * 1e9, 10, 20, 50, 100, 250, 500, DMPS.d_m[-1] * 1e9])
    
    axs[1].set_ylabel('Particle diameter (nm)')
    axs[1].set_xlabel('Time')
    axs[1].set_title(f'Estimate uncertainty (width of the {CI_coverage} % credible intervals)')
                
    cbar = fig.colorbar(im, ax=axs[1],
                         label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$')
    
    fig.tight_layout()
    plt.show()
    
    
    # Variance reduction plot
    prior_variance = np.diag(prior['covariance'])
    post_variance = inv_dataset.posterior_variance()
    VR = np.log10(post_variance / prior_variance)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, num=100, clear=True)
    im = ax.pcolormesh(*np.meshgrid(datetimes, DMPS.d_m * 1e9), VR.T,
                       cmap='Blues_r')
    cbar = fig.colorbar(im, ax=ax, label=r'$\log_{10}(\sigma_\mathrm{post}^2 / \sigma_\mathrm{prior}^2)$')
    
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.0f'))
    ax.set_yticks([DMPS.d_m[0] * 1e9, 10, 20, 50, 100, 250, 500, DMPS.d_m[-1] * 1e9])
    ax.set_ylabel('Particle diameter (nm)')
    ax.set_xlabel('Time')
    ax.set_title('log variance reduction')
    
    fig.tight_layout()
    plt.show()
    
    
    # Example measurements
    fig = plt.figure(num=12, clear=True)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.05, 1])  # middle row is a gap for a 
    axs = np.empty((2, 2), dtype=object)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        ]
    
    idx_1 = 10
    idx_2 = 100
    # Convert numpy datetimes to Python datetimes for easier formatting
    datetime_1 = inv_dataset.datetimes[idx_1].astype('datetime64[s]').item()
    datetime_2 = inv_dataset.datetimes[idx_2].astype('datetime64[s]').item()
    
    plot_posterior_summary(axs[0], inv_dataset.results[idx_1], CI_coverage)
    axs[0].set_yscale('linear')
    axs[0].set_xlim([DMPS.d_m[0] * 1e9, DMPS.d_m[-1] * 1e9])
    axs[0].grid('on')
    axs[0].legend()
    axs[0].set_title(
        f'Size distribution on {datetime_1.date()} at {datetime_1.time()}',
        loc='center'
        )
    axs[0].set_ylim(0, 4000)
    
    
    plot_posterior_summary(axs[2], inv_dataset.results[idx_2], CI_coverage)
    axs[2].set_yscale('linear')
    axs[2].set_xlim([DMPS.d_m[0] * 1e9, DMPS.d_m[-1] * 1e9])
    axs[2].grid('on')
    axs[2].legend()
    axs[2].set_title(
        f'Size distribution on {datetime_2.date()} at {datetime_2.time()}',
        loc='center'
        )
    axs[2].set_ylim(0, 2000)
    
    Ntot_samples_1 = inv_dataset.results[idx_1].Ntot_samples()
    Ntot_samples_2 = inv_dataset.results[idx_2].Ntot_samples()
    plot_Ntot_histogram(axs[1], Ntot_samples_1)
    plot_Ntot_histogram(axs[3], Ntot_samples_2)
    
    
    line = Line2D([0.075, 0.95], [0.50, 0.50], transform=fig.transFigure,
                  color='black', linewidth=4)
    fig.add_artist(line)

    
    
    fig.tight_layout()
    plt.show()

