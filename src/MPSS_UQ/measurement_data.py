# -*- coding: utf-8 -*-

import numpy as np
import warnings

from MPSS_UQ.particlesizers import DifferentialMobilityParticleSizer


class Measurement:
    ''' A class to store a _single_ measurement. This is given to the inversion routines.
    kwargs can be used to give optional input values, such as the true N in case of simulated
    data (see allowed_optional_keys).
    '''
    allowed_optional_keys = {'N_true', 'scenario', 'binwidth', 'd_m_true'}
    
    def __init__(self, datetime, d_m_data, MPSS_output, output_type, temperature, pressure,
                 **kwargs):
        
        self.datetime = datetime
        self.d_m_data = d_m_data
        self.output = MPSS_output
        self.output_type = output_type
        self.temperature = temperature
        self.pressure = pressure
        
        for key, value in kwargs.items():
            if key in self.allowed_optional_keys:
                setattr(self, key, value)
            else:
                raise ValueError(f'Unexpected keyword argument: {key}')
    
    
    def compute_noise_statistics(self):
        if self.output_type == 'counts':
            
            if np.any(self.output < 0):
                warnings.warn( "Observed negative counts. Clipping them to zero for noise " +
                              "estimation purposes.", UserWarning)
            
            self.noise_cov = np.clip(self.output, 0, np.inf).astype(np.float64)
            self.noise_cov += (2 + np.clip(self.output, 0, np.inf) * 0.01)**2
            self.inv_noise_cov = 1 / self.noise_cov
            
            # Matrix square root of the inverse noise covariance
            self.noise_L = np.sqrt(self.inv_noise_cov)
            
            # Make them matrices
            self.noise_cov = np.diag(self.noise_cov)
            self.inv_noise_cov = np.diag(self.inv_noise_cov)
            self.noise_L = np.diag(self.noise_L)
            
        elif self.output_type == 'concentration':
            # TODO
            raise TypeError('Concentration output has not been implemented yet. It is ' +
                            'recommended the measurement output is given as counts.')
        else:
            raise ValueError('Unknown CPC output type')
    
    
    def preprocess(self):
        self.compute_noise_statistics()


class MeasurementDataset:
    
    def __init__(self, datetimes, d_m_data, MPSS_outputs, output_type, temperatures, pressures):
        
        self.datetimes = datetimes
        self.d_m_data = d_m_data
        self.output_vector = MPSS_outputs
        self.output_type = output_type
        self.temperatures = temperatures
        self.pressures = pressures
    
    
    def __len__(self):
        return len(self.output_vector)
    
    
    def __getitem__(self, idx):
        
        measurement = Measurement(self.datetimes[idx],
                                  self.d_m_data,
                                  self.output_vector[idx],
                                  self.output_type,
                                  self.temperatures[idx],
                                  self.pressures[idx],
                                  )
        
        measurement.preprocess()
        
        return measurement


def measurement_loader(dataset):
    
    for i in range(len(dataset)):
        yield dataset[i]


def generate_DMPS_measurement(DMPS_prop,
                              scenario,
                              pos_ion_mobility=1.35e-4,
                              neg_ion_mobility=1.60e-4,
                              ):
    ''' Simulate measurement data using a DMPS model.
    Input:
        DMPS_prop - the DMPS properties dictionary
        scenario - name of the true PSD we simulate
    Predefined PSD scenarios are:
        Urban, Marine, Rural, Remote continental, Free troposphere, Polar, Desert.
    '''
    
    # Mobility diameters that are used to represent the true PSD
    d_m = np.geomspace(1e-9, 2500e-9, num=500)

    # Which bipolar charging model to use
    DMPS_prop['charging_model'] = 'LYF-interp'

    # Maximum considered number of multiple charges
    DMPS_prop['max_charge'] = 8

    # Create the data generating DMPS object
    DMPS = DifferentialMobilityParticleSizer(DMPS_prop, inversion_grid=d_m)
    
    # Set the ambient temperature and pressure
    temperature = 293.15  # [K]
    pressure = 101325  # [Pa]
    
    DMPS.set_operating_conditions(temperature, pressure)

    # Set charger ion properties for the measurement
    DMPS.set_charger_properties(pos_ion_mobility, neg_ion_mobility)
    
    # from MPSS_UQ.plotfunctions import plot_system_matrix
    # plot_system_matrix(DMPS, title='data generation')
    
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
    elif scenario == 'test':
        dN_dlogdp_true = 1000 * np.ones_like(DMPS.d_m)
    
    elif scenario == 'test2':
        dN_dlogdp_true = 5000 * np.linspace(1, 1e-5, len(DMPS.d_m))
    
    elif scenario == 'test3':
        dN_dlogdp_true = 5000 * np.ones_like(DMPS.d_m)
        dN_dlogdp_true[int(len(DMPS.d_m) / 2):] = 1e-5
        
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
    rng = np.random.default_rng()#seed=1)
    DMPS_output = rng.poisson(lam=DMPS_output_noiseless).astype(np.float64)
    noise_std = 2 + 0.01 * DMPS_output_noiseless
    DMPS_output += noise_std * rng.normal(loc=0.0, scale=1.0,
                                          size=DMPS_output.shape)
    
    # Create a Measurement object
    measurement = Measurement(None,
                              DMPS.d_m_data,
                              DMPS_output,
                              'counts',
                              temperature,
                              pressure,
                              N_true=N_true,
                              scenario=scenario,
                              d_m_true=d_m,
                              )
    
    measurement.preprocess()
    
    return measurement


def lognormal_distribution(d_m, N, median, log_std):
    ''' Returns the lognormal distribution with the specified properties. '''
    
    return N / (np.sqrt(2 * np.pi) * log_std) * np.exp(
        - 0.5 * (np.log10(d_m) - np.log10(median))**2 / (log_std)**2
        )


def compute_true_Ntot_in_range(measurement, d_m_inv):
    """
    Compute the total particle number concentration (Ntot) in the inversion size range
    from the true PSD used to generate synthetic data.

    Parameters
    ----------
    measurement :
        A MeasurementData object with the truth known (simulated data).
    d_m_inv : ndarrau
        Diameters used in the inversion.

    Returns
    -------
    Ntot_true : float
        Total particle number concentration of the true measurement in the inversion size range.
    """
    
    if not hasattr(measurement, 'N_true'):
        raise ValueError("Did not find 'N_true' in 'measurement'. Make sure 'measurement' " +
                         "includes the true values.")
    if not hasattr(measurement, 'd_m_true'):
        raise ValueError("Did not find 'N_true' in 'measurement'. Make sure 'measurement' " +
                         "includes the true values.")
    
    # Calculate the true Ntot but only for sizes we consider in the inversion.
    # To compare them accurately, we need to consider the bin widths. Assuming here that
    # the widths of the data generating bins are smaller than that of the inversion bins.
    binwidth_inv_log10 = np.log10(d_m_inv[1]) - np.log10(d_m_inv[0])
    binwidth_fwd_log10 = np.log10(measurement.d_m_true[1]) - np.log10(measurement.d_m_true[0])
    inv_edge_smallest = 10**(np.log10(d_m_inv[0]) - 0.5 * binwidth_inv_log10)
    inv_edge_largest = 10**(np.log10(d_m_inv[-1]) + 0.5 * binwidth_inv_log10)
    fwd_edges = 10**(np.concatenate(
        (np.log10(measurement.d_m_true) - 0.5 * binwidth_fwd_log10,
         [np.log10(measurement.d_m_true[-1]) + 0.5 * binwidth_fwd_log10]
         )))

    # Find first those data generating bins that are fully inside the inverted interval
    # Do this by comparing the bin edges
    if fwd_edges[-1] < inv_edge_largest or fwd_edges[0] > inv_edge_smallest:
        warnings.warn(
            "Size range of the inversion bins is wider than that of the data generating bins. " +
            "Comparing Ntots may not be reliable.", UserWarning)
    # left edge of bin k = fwd_edges[k]
    # right edge of bin k = fwd_edges[k+1]
    idx1 = np.where(fwd_edges >= inv_edge_smallest)[0][0]
    idx2 = np.where(fwd_edges <= inv_edge_largest)[0][-1]
    Ntot_true = np.sum(measurement.N_true[idx1 : idx2 + 1])
    # Include possible partial contributions from below the first and above the last bin
    if fwd_edges[idx1] > inv_edge_smallest:
        if idx1 > 0:
            fraction = (fwd_edges[idx1] - inv_edge_smallest) / (
                fwd_edges[idx1] - fwd_edges[idx1 - 1])
            Ntot_true += fraction * measurement.N_true[idx1 - 1]
    if fwd_edges[idx2] < inv_edge_largest:
        if idx2 < len(measurement.N_true):
            fraction = (inv_edge_largest - fwd_edges[idx2]) / (
                fwd_edges[idx2 + 1] - fwd_edges[idx2])
            Ntot_true += fraction * measurement.N_true[idx2]
    
    
    return Ntot_true
