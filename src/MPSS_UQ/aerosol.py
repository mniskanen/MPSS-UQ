# -*- coding: utf-8 -*-
"""
Basic aerosol microphysics utilities for spherical particles in air.
"""

import numpy as np


BOLTZMANN_CONSTANT = 1.380649e-23  # Boltzmann constant
ELECTRON_CHARGE = 1.602176634e-19  # Elementary charge
VACUUM_PERMITTIVITY = 8.8541878188e-12
AIR_RELATIVE_PERMITTIVITY = 1.00058986  # i.e., its dielectric constant


def dynamic_viscosity(temperature):
    """ Dynamic viscosity of air given by the Sutherland formula. """
    return 1.458e-6 * temperature**1.5 / (temperature + 110.4)


def mean_free_path_air(temperature, pressure):
    """ Calculate the mean free path of an air molecule for a given temperature.
    Assume that the air molecules consist of nitrogen only.
    """
    
    # Diameter of a nitrogen molecule (adjusted so that we get a mean free path of 68 nm for
    # temperature 20 C)
    d = 3.64e-10
    
    return BOLTZMANN_CONSTANT * temperature / (np.sqrt(2) * np.pi * d**2 * pressure)


def cunningham(Kn):
    """Compute the Cunningham slip correction factor for given Knudsen numbers Kn,
    defined as Kn = 2 \lambda / d.
    """
    return 1 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))


def particle_diffusivity(d_m, temperature, pressure):
    """ Computes the particle diffusion coefficient. """
    Kn = 2 * mean_free_path_air(temperature, pressure) / d_m
    mu = dynamic_viscosity(temperature)
    Cc = cunningham(Kn)
    
    return BOLTZMANN_CONSTANT * temperature / (3 * np.pi * mu * d_m) * Cc


def electrical_mobility(d_m, temperature, pressure, particle_charge):
    ''' Compute electrical mobility for spherical particles from the particle mobility
    diameter.
    particle_charge is given as the number of elementary charges on the particle.
    '''
    
    Kn = 2 * mean_free_path_air(temperature, pressure) / d_m
    eta = dynamic_viscosity(temperature)
    
    return np.abs(particle_charge) * ELECTRON_CHARGE * cunningham(Kn) / (3 * np.pi * eta * d_m)