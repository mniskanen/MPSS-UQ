# -*- coding: utf-8 -*-

import numpy as np

from MPSS_UQ.particlesizers import DifferentialMobilityParticleSizer
from MPSS_UQ.measurement_data import Measurement


def lognormal_distribution(d_m, N, median, log_std):
    ''' Returns the lognormal distribution with the specified properties. '''
    
    return N / (np.sqrt(2 * np.pi) * log_std) * np.exp(
        - 0.5 * (np.log10(d_m) - np.log10(median))**2 / (log_std)**2
        )


def generate_DMPS_measurement(DMPS_prop, scenario):
    ''' Simulate measurement data using a DMPS model.
    Input:
        DMPS_prop - the DMPS properties dictionary
        scenario - name of the true PSD we simulate
    Predefined PSD scenarios are:
        Urban, Marine, Rural, Remote continental, Free troposphere, Polar, Desert.
    '''
    
    # Mobility diameters that are used to represent the true PSD
    DMPS_prop['d_m'] = np.geomspace(1e-9, 2500e-9, num=500)

    # Which bipolar charging model to use
    DMPS_prop['charging_model'] = 'LYF-interp-fast'

    # Maximum considered number of multiple charges
    DMPS_prop['max_charge'] = 8

    # Create the data generating DMPS object
    DMPS = DifferentialMobilityParticleSizer(DMPS_prop)
    
    # Set the ambient temperature and pressure
    temperature = 293.15  # [K]
    pressure = 101325  # [Pa]
    
    DMPS.set_operating_conditions(temperature, pressure)

    # Set charger ion properties for the measurement (if using the LYF model)
    pos_ion_mobility = 1.4e-4
    neg_ion_mobility = 1.9e-4
    # ion_ratio = 1.0
    DMPS.set_charger_properties(pos_ion_mobility, neg_ion_mobility)#, ion_ratio)
    
    # from MPSS_UQ.plotfunctions import plot_system_matrix
    # plot_system_matrix(DMPS)
    
    # Generate a synthetic particle size distribution (based on Seinfeld & Pandis Table 8.3)
    # Define parameters for each scenario, these define a sum of three lognormal modes
    scenario_params = {
        'Urban': [(7100, 11.7e-9, 0.232),
                  (6320, 37.3e-9, 0.250),
                  (960, 151e-9, 0.204)],
        
        'Marine': [(133, 8e-9, 0.657),
                   (66.6, 266e-9, 0.210),
                   (3.1, 580e-9, 0.396)],
        
        'Rural': [(6650, 15e-9, 0.225),
                  (147, 54e-9, 0.557),
                  (1990, 84e-9, 0.266)],
        
        'Remote continental': [(3200, 20e-9, 0.161),
                               (2900, 116e-9, 0.217),
                               (0.3, 1800e-9, 0.380)],
        
        'Free troposphere': [(129, 7e-9, 0.645),
                             (59.7, 250e-9, 0.253),
                             (63.5, 520e-9, 0.425)],
        
        'Polar': [(21.7, 138e-9, 0.245),
                  (0.186, 750e-9, 0.300),
                  (3e-4, 8600e-9, 0.291)],
        
        'Desert': [(726, 2e-9, 0.247),
                   (114, 38e-9, 0.770),
                   (0.178, 21600e-9, 0.438)],
        }
    
    
    # Input the parameters into lognormal_distribution()
    if scenario in scenario_params:
        dN_dlogdp_true = sum(lognormal_distribution(DMPS.d_m, N, median, log_std)
                             for N, median, log_std in scenario_params[scenario]
                             )
    else:
        valid_scenarios = ', '.join(scenario_params.keys())
        raise ValueError('Undefined PSD scenario. Use one of the following: ' +
                         f'{valid_scenarios}')
    
    # Transform to a bin representation by multiplying by the log bin width.
    # Assume log width of all bins is the same.
    binwidth = np.log10(DMPS.d_m[1]) - np.log10(DMPS.d_m[0])
    N_true = dN_dlogdp_true * binwidth
    
    # Generate a DMA observation
    DMPS_output_noiseless = DMPS.forward_model(np.log10(N_true))
    
    # Add noise
    rng = np.random.default_rng(seed=1)
    DMPS_output = rng.poisson(lam=DMPS_output_noiseless)
    
    # Create a Measurement object
    measurement = Measurement(None,
                              DMPS.d_m_data,
                              DMPS_output,
                              'counts',
                              temperature,
                              pressure,
                              N_true=N_true,
                              scenario=scenario,
                              d_m_truth=DMPS_prop['d_m'],
                              )
    
    measurement.preprocess()
    
    return measurement
