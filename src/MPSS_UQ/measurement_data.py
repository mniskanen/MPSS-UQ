# -*- coding: utf-8 -*-

import numpy as np


class Measurement:
    ''' A class to store a _single_ measurement. This is given to the inversion routines.
    kwargs can be used to give optional input values, such as the true N in case of simulated
    data (see allowed_optional_keys).
    '''
    allowed_optional_keys = {'N_true', 'n_true', 'scenario', 'binwidth', 'd_m_truth'}
    
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
            self.noise_cov = np.diag(np.clip(self.output, 1, np.inf))
            self.inv_noise_cov = np.diag(1 / np.diag(self.noise_cov))
            
            # Matrix square root of the inverse noise covariance
            self.noise_L = np.diag(np.sqrt(np.diag(self.inv_noise_cov)))
            
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