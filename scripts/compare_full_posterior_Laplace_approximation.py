# -*- coding: utf-8 -*-

import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from MPSS_UQ.particlesizers import MobilityParticleSizeSpectrometer
from MPSS_UQ.inversion import invert_psd
from MPSS_UQ.inversion_results import highest_density_interval
from MPSS_UQ.measurement_data import generate_DMPS_measurement
from MPSS_UQ.plotfunctions import plot_psd, plot_posterior_summary, plot_Ntot_histogram, plot_datafit


''' This script compares the Laplace approximation to the posterior and the true posterior in
a few cases of synthetic data. '''


def plot_comparison(ax, DMPS, measurement, result_Laplace, result_MCMC, title, ylims):

    binwidth = np.log10(DMPS.d_m[1]) - np.log10(DMPS.d_m[0])    
    binwidth_m = np.log10(measurement.d_m_true[1]) - np.log10(measurement.d_m_true[0])
    
    # Highest density intervals for MCMC
    CI_lower = np.zeros(DMPS.d_m.shape[0])
    CI_upper = np.zeros_like(CI_lower)
    for k in range(DMPS.d_m.shape[0]):
        CI_lower[k], CI_upper[k] = highest_density_interval(result_MCMC.post_samples[:, k],
                                                            95 / 100
                                                            )
    
    # HDI for Laplace
    sigma = np.sqrt(np.sum(result_Laplace.post_covL_log10**2, axis=1))
    k = norm.ppf(0.5 + 95 / 100 / 2)
    
    CI_lower_MAP = 10**(result_Laplace.post_mean_log10 - k * sigma)
    CI_upper_MAP = 10**(result_Laplace.post_mean_log10 + k * sigma)
    
    ax.semilogx(DMPS.d_m, np.mean(result_MCMC.post_samples, axis=0) / binwidth, 'C0',
                label='CM estimate (MCMC)'
                )
    ax.semilogx(DMPS.d_m, 10**result_Laplace.post_mean_log10 / binwidth, 'C1',
                label='MAP estimate'
                )
    ax.semilogx(measurement.d_m_true, measurement.N_true / binwidth_m, '--k',
                label='Truth'
                )
    
    ax.fill_between(DMPS.d_m,
                    CI_upper / binwidth,
                    CI_lower / binwidth,
                    alpha=0.25,
                    facecolor='C0',
                    label='95 % CI (MCMC)'
                    )
    ax.fill_between(DMPS.d_m,
                    CI_upper_MAP / binwidth,
                    CI_lower_MAP / binwidth,
                    alpha=0.25,
                    facecolor='C1',
                    label='95 % CI (Laplace)'
                    )
    
    ax.set_xlim([result_MCMC.d_m[0], result_MCMC.d_m[-1]])
    ax.set_ylim(ylims)
    ax.legend()
    ax.set_ylabel(r'dN / d$\log$d$_m$')
    ax.set_xlabel('Diameter (m)')
    ax.set_title('Posterior comparison' + title)
    plt.pause(1.0)


def plot_histogram_comparison(ax, mcmc_samples, MAP, post_cov, title=None):
    
    # Histogram of MCMC samples
    ax.hist(mcmc_samples, bins=50, density=True, alpha=0.6, color='skyblue', label='MCMC samples')
    
    post_std = np.sqrt(post_cov)
    # First find x limits that cover all MCMC samples and at least +- 2 * Laplace approximation
    # standard deviations
    # x_min = min(min(mcmc_samples), MAP - 2 * post_std)
    # x_max = max(max(mcmc_samples), MAP + 2 * post_std)
    # # Make the Laplace approximation plot symmetric around the MAP
    # abs_max = max(abs(MAP - x_min), abs(MAP - x_max))
    x_min = MAP - 4 * post_std#abs_max
    x_max = MAP + 4 * post_std#abs_max
    x = np.linspace(x_min, x_max, 500)
    
    # Laplace approximation
    laplace_pdf = norm.pdf(x, loc=MAP, scale=np.sqrt(post_cov))
    ax.plot(x, laplace_pdf, '-r', lw=2, label='Laplace approximation')
    
    # Labels and legend
    ax.set_xlabel(r'$\log_{10}(N)$')
    ax.set_ylabel('Density')
    ax.set_title('Marginal posterior' + title)
    ax.legend()


def compare_posterior_representations(ax1, ax2, ax3, ax4):
    
    with open("../examples/DMPS_properties.yaml", "r") as f:
        DMPS_prop = yaml.safe_load(f)
    
    DMPS_prop = DMPS_prop['UEF-A20']
    DMPS_prop['d_m_data'] = np.geomspace(10e-9, 800e-9, num=30)
    DMPS_prop['charging_model'] = 'LYF-interp'
    DMPS_prop['max_charge'] = 10
    
    DMPS = MobilityParticleSizeSpectrometer(DMPS_prop, n_bins=70)
    
    pos_ion_mob = 1.35e-4
    neg_ion_mob = 1.60e-4
    
    scenario_1 = 'Desert'
    scenario_2 = 'Urban'
    
    # Test 1
    measurement = generate_DMPS_measurement(DMPS_prop.copy(),
                                            scenario=scenario_1,
                                            pos_ion_mobility=pos_ion_mob,
                                            neg_ion_mobility=neg_ion_mob,
                                            rng_seed=10
                                            )
    
    DMPS.set_operating_conditions(measurement.temperature, measurement.pressure)
    DMPS.set_charger_properties(pos_ion_mob, neg_ion_mob)
    
    # Laplace approximation
    result_Laplace = invert_psd(DMPS, measurement)
    
    # Sampled posterior
    result_MCMC = invert_psd(DMPS, measurement, sample_posterior=True, num_samples=500000)
    
    plot_comparison(ax1, DMPS, measurement, result_Laplace, result_MCMC,
                    ', ' + scenario_1 + ' scenario', [0, 300])
    
    bb = 1  # bin number
    post_cov = result_Laplace.post_covL_log10 @ result_Laplace.post_covL_log10.T
    plot_histogram_comparison(ax3,
                              np.log10(result_MCMC.post_samples[:, bb]),
                              result_Laplace.post_mean_log10[bb],
                              post_cov[bb, bb],
                              title=f' at {result_Laplace.d_m[bb] * 1e9 : .1f} nm, '
                              + scenario_1 + ' scenario'
                              )
    
    
    
    # Test 2
    measurement = generate_DMPS_measurement(DMPS_prop.copy(),
                                            scenario=scenario_2,
                                            pos_ion_mobility=pos_ion_mob,
                                            neg_ion_mobility=neg_ion_mob,
                                            rng_seed=10
                                            )
    
    DMPS.set_operating_conditions(measurement.temperature, measurement.pressure)
    DMPS.set_charger_properties(pos_ion_mob, neg_ion_mob)
    
    # Laplace approximation
    result_Laplace = invert_psd(DMPS, measurement)
    
    # Sampled posterior
    result_MCMC = invert_psd(DMPS, measurement, sample_posterior=True, num_samples=500000)
    
    plot_comparison(ax2, DMPS, measurement, result_Laplace, result_MCMC,
                    ', ' + scenario_2 + ' scenario', [0, 20000])
    
    post_cov = result_Laplace.post_covL_log10 @ result_Laplace.post_covL_log10.T
    plot_histogram_comparison(ax4,
                              np.log10(result_MCMC.post_samples[:, bb]),
                              result_Laplace.post_mean_log10[bb],
                              post_cov[bb, bb],
                              title=f' at {result_Laplace.d_m[bb] * 1e9 : .1f} nm, '
                              + scenario_2 + ' scenario'
                              )


if __name__ == '__main__':
    fig, axs = plt.subplots(ncols=2, nrows=2, num=99, clear=True)    
    compare_posterior_representations(axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1])
