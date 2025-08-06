# -*- coding: utf-8 -*-

import numpy as np


def lognormal_distribution(d_m, N, median, log_std):
    ''' Returns the lognormal distribution with the specified properties. '''
    
    return N / (np.sqrt(2 * np.pi) * log_std) * np.exp(
        - 0.5 * (np.log10(d_m) - np.log10(median))**2 / (log_std)**2
        )


def generate_DMPS_measurement(DMPS, scenario='Urban'):
    ''' Simulate measurement data using the DMPS model.
    Predefined PSD scenarios are:
        Urban, Marine, Rural
    '''
    
    # from MPSS_UQ.plotfunctions import plot_system_matrix
    # plot_system_matrix(DMPS)
    
    # Generate a synthetic particle size distribution (based on Seinfeld & Pandis Table 8.3)
    if scenario == 'Urban':
        n_true = (lognormal_distribution(DMPS.d_m, 7100, 11.7e-9, 0.232)
                  + lognormal_distribution(DMPS.d_m, 6320, 37.3e-9, 0.250)
                  + lognormal_distribution(DMPS.d_m, 960, 151e-9, 0.204))
    
    elif scenario == 'Marine':
        n_true = (lognormal_distribution(DMPS.d_m, 133, 8e-9, 0.657)
                  + lognormal_distribution(DMPS.d_m, 66.6, 266e-9, 0.210)
                  + lognormal_distribution(DMPS.d_m, 3.1, 580e-9, 0.396))
    
    elif scenario == 'Rural':
        n_true = (lognormal_distribution(DMPS.d_m, 6650, 15e-9, 0.225)
                  + lognormal_distribution(DMPS.d_m, 147, 54e-9, 0.557)
                  + lognormal_distribution(DMPS.d_m, 1990, 84e-9, 0.266))
    
    elif scenario == 'Remote continental':
        n_true = (lognormal_distribution(DMPS.d_m, 3200, 20e-9, 0.161)
                  + lognormal_distribution(DMPS.d_m, 2900, 116e-9, 0.217)
                  + lognormal_distribution(DMPS.d_m, 0.3, 1800e-9, 0.380))
    
    elif scenario == 'Free troposphere':
        n_true = (lognormal_distribution(DMPS.d_m, 129, 7e-9, 0.645)
                  + lognormal_distribution(DMPS.d_m, 59.7, 250e-9, 0.253)
                  + lognormal_distribution(DMPS.d_m, 63.5, 520e-9, 0.425))
    
    elif scenario == 'Polar':
        n_true = (lognormal_distribution(DMPS.d_m, 21.7, 138e-9, 0.245)
                  + lognormal_distribution(DMPS.d_m, 0.186, 750e-9, 0.300)
                  + lognormal_distribution(DMPS.d_m, 3e-4, 8600e-9, 0.291))
    
    elif scenario == 'Desert':
        n_true = (lognormal_distribution(DMPS.d_m, 726, 2e-9, 0.247)
                  + lognormal_distribution(DMPS.d_m, 114, 38e-9, 0.770)
                  + lognormal_distribution(DMPS.d_m, 0.178, 21600e-9, 0.438))
    
    else:
        raise ValueError('Undefined PSD scenario.')
    
    # The above true values are given in terms of the PSD density function.
    # To transform to a bin representation let's multiply the bin midpoint by the bin width.
    # Assume log width of all bins is the same.
    binwidth = np.log10(DMPS.d_m[1]) - np.log10(DMPS.d_m[0])
    N_true = n_true * binwidth
    
    # Generate a DMA observation
    DMPS_output_noiseless = DMPS.forward_model(np.log10(N_true))
    
    # Add noise
    rng = np.random.default_rng(seed=1)
    DMPS_output = rng.poisson(lam=DMPS_output_noiseless)
    
    # Collect into a dictionary
    measurement = {
        'scenario' : scenario,
        'output' : DMPS_output,  # can be either counts or concentration depending on the settings
        'output_noiseless' : DMPS_output_noiseless,
        'N_true' : N_true,
        'n_true' : n_true,
        'binwidth' : binwidth,
        'd_m' : DMPS.d_m,
        'd_m_data' : DMPS.d_m_data,
        }
    
    # Add noise parameters
    measurement['noise_cov'] = np.diag(np.clip(measurement['output'], 1, np.inf))
    measurement['inv_noise_cov'] = np.diag(1 / np.diag(measurement['noise_cov']))
    measurement['noise_L'] = np.diag(np.sqrt(np.diag(measurement['inv_noise_cov'])))
    
    return measurement
