# -*- coding: utf-8 -*-

import numpy as np
import importlib.resources as resources

from scipy.special import erf
# from time import perf_counter

from MPSS_UQ.chargingmodels import (LYFChargingModel,
                                    LYFFluxInterpolator,
                                    WiedensohlerChargingModel,
                                    )


BOLTZMANN_CONSTANT = 1.380649e-23  # Boltzmann constant
EL_CHARGE = 1.602176634e-19  # Elementary charge

lpm_to_m3s = 1e-3 / 60  # Liters per minute to m3 per second conversion factor


# TODO: move these functions somewhere sensible
def dynamic_viscosity(temperature):
    """ Dynamic viscosity of air given by the Sutherland formula. """
    return 1.458e-6 * temperature**1.5 / (temperature + 110.4)


def mean_free_path_air(temperature):
    """ Calculate the mean free path of an air molecule for a given temperature. Assume that
    the air molecules consist of nitrogen only.
    """
    # Air pressure (Pa)
    p = 101325
    
    # Diameter of a nitrogen molecule (adjusted so that we get a mean free path of 68 nm for
    # temperature 20 C)
    d = 3.64e-10
    # d = 3.711e-10
    
    return BOLTZMANN_CONSTANT * temperature / (np.sqrt(2) * np.pi * d**2 * p)

def cunningham(Kn):
    """Compute the Cunningham slip correction factor for given Knudsen numbers Kn."""
    return 1 + Kn * (1.257 + 0.4 * np.exp(-1.1 / Kn))


class DifferentialMobilityParticleSizer():
    
    def __init__(self, properties):
        
        # Mobility diameters the device has been set to measure (corresponding to the
        # voltages measured)
        self.d_m_data = properties['d_m_data']
        
        # Diameters at which we want to calculate the solution of the inverse problem
        self.d_m = properties['d_m']
        
        self.Qsh = properties['Qsh']  # Sheath flow (liters per minute)
        self.Qe = properties['Qe']  # Exhaust flow (lpm)
        self.Qa = properties['Qa']  # Aerosol flow (lpm)
        self.Qc = properties['Qc']  # Classified sample flow (lpm)
        # Convert flows from lpm to m^3/s
        self.Qsh *= lpm_to_m3s
        self.Qe *= lpm_to_m3s
        self.Qa *= lpm_to_m3s
        self.Qc *= lpm_to_m3s
        
        self.temperature = properties['T']  # Internal temperature (K)
        self.pressure = properties['P']  # Internal pressure (Pa)
        self.R1 = properties['R1']  # Inner radius of the DMA (m)
        self.R2 = properties['R2']  # Outer radius of the DMA (m)
        self.length = properties['L']  # Length of the DMA (m)
        
        # Sign of center electrode voltage compared to the outer electrode.
        self.center_voltage_sign = properties['center_voltage_sign']
        
        if self.center_voltage_sign == 'negative':
            # Classify positively charged particles
            self.particle_charge_sign = 'positive'
        
        elif self.center_voltage_sign == 'positive':
            # Classify negatively charged particles
            self.particle_charge_sign = 'negative'
        
        # Max considered nbr of charges on a particle (always a positive number)
        self.max_charge = properties['max_charge']
        
        assert type(self.max_charge) == int \
            and self.max_charge > 0, 'The maximum charge has to be a positive integer'
        
        # Amounts of charge (negative or positive) considered on a particle
        self.charges = np.arange(1, self.max_charge + 1)
        if self.particle_charge_sign == 'negative':
            self.charges = -self.charges
            self.charges.sort()  # Charges need to be in the right order
        
        # Some derived values used later on
        self.compute_DMA_characteristic_values()
        
        # Mobilities we want to classify
        self.Z_targets = self.compute_electrical_mobility(self.d_m_data, 1)
        
        # Particle charging model
        self.initialize_charging_model(properties['charging_model'])
        
        # CPC
        self.CPC = CondensationParticleCounter(self.d_m_data, properties)
        
        self.system_matrix = np.zeros((self.d_m_data.shape[0], self.d_m.shape[0]))
        
        self.transfer_function = self.compute_transfer_function()
        self.charging_probability = self.compute_charging_probability()
        self.penetration_efficiency = self.compute_penetration_efficiency()
        self.update_system_matrix()
    
    
    def forward_model(self, log10_N):
        ''' Run a particle population N through the measurement system and output counts. '''
        
        return self.system_matrix @ 10**log10_N
    
    
    def update_charger_ion_properties(self, pos_ion_mobility, neg_ion_mobility, ion_ratio):
        ''' Calculate the charging probability with the inputted ion mobilities and ratio,
        and update the DMA system matrix.
        '''
        
        if self.charging_model_name == 'Wiedensohler':
            raise Exception('Cannot change charger properties with the Wiedensohler model')
        
        self.charging_probability = self.compute_charging_probability(pos_ion_mobility,
                                                                      neg_ion_mobility,
                                                                      ion_ratio,
                                                                      )
        
        self.update_system_matrix()
    
    
    def update_system_matrix(self, charging_probability=None):
        ''' Calculate the system matrix, a matrix that models the whole DMA+CPC system
        including the transfer function, and charging, penetration, and counting efficiencies.
        Using this function requires that the individual parts of the system function have been
        calculated before. This is done so that the charging probability can be updated without
        having to compute the other parts of the system function again.
        
        Input:
        charging_probability, defaults to None but given as an option if there is a need to update
                              the charging model from outside the class.
        '''
        
        if charging_probability is not None:
            self.charging_probability = charging_probability
        
        self.system_matrix *= 0
        
        # Sum the transfer function over each considered charge and multiply with
        # the charging probability
        for c_idx, charge in enumerate(self.charges):
            self.system_matrix += (
                self.transfer_function[c_idx] * self.charging_probability[c_idx]
                )
        
        self.system_matrix *= self.CPC.counting_efficiency[:, np.newaxis]
        self.system_matrix *= self.penetration_efficiency
        
        # Change output from concentration to counts
        if self.CPC.output_type == 'counts':
            self.system_matrix *= self.CPC.sampled_volume
    
    
    def compute_DMA_characteristic_values(self):
        ''' Compute some values that characterize a DMA that are used later on in other
        calculations. '''
        
        # Flow rate ratios
        self.beta = (self.Qa + self.Qc) / (self.Qsh + self.Qe)
        self.delta = (self.Qc - self.Qa) / (self.Qc + self.Qa)
        
        # Dimensionless parameter G, assuming uniform plug flow
        gamma = (self.R1 / self.R2)**2
        I = 0.5 * (1 + gamma)
        kappa = self.length * self.R2 / (self.R2**2 - self.R1**2)
        
        self.G = 4 * (1 + self.beta)**2 / (1 - gamma) * (I + (2 * (1 + self.beta) * kappa)**-2)
    
    
    def compute_penetration_efficiency(self):
        ''' Note: this is just an example and should be modified to the DMPS used. '''
        
        a = 3.66
        b = 0.2672
        c = 0.10079
        
        Kn = mean_free_path_air(self.temperature) / (self.d_m_data * 0.5)
        mu = dynamic_viscosity(self.temperature)
        Cc = cunningham(Kn)
        
        # Diffusion coefficient
        D = BOLTZMANN_CONSTANT * self.temperature / (3 * np.pi * mu * self.d_m_data) * Cc
        
        L_eff = 4.6
        
        # Particle diffusion coefficient
        tau = np.pi * D * L_eff / self.Qa
        
        # Sherwood number for laminar flow
        Sh = a + b / (tau + c * tau**(1 / 3))
        
        return np.exp(-tau * Sh)[:, np.newaxis]
    
    
    def _tf_eps(self, y):
        ''' The epsilon function for computing the diffusive transfer function. '''
        
        return y * erf(y) + 1 / np.sqrt(np.pi) * np.exp(-y**2)
    
    
    def _tf_element_nondiffusive(self, Z, Z_target):
        ''' Compute a single element of the DMA transfer function assuming no diffusion. '''
        Z_ratio = Z / Z_target
        
        term1 = (Z_ratio + self.beta - 1) / (self.beta - self.beta * self.delta)
        term2 = (1 + self.beta - Z_ratio ) / (self.beta - self.beta * self.delta)
        
        TF = np.max((
            np.zeros_like(term1), np.min((np.ones_like(term1), term1, term2), axis=0)
            ), axis=0)
        
        return TF
    
    
    def _tf_element_diffusive(self, Z, Z_target, charge):
        ''' Compute a single element of the DMA transfer function with diffusion.
        Computed after Stolzenburg 1988 'An ultrafine aerosol size distribution system'.
        '''
        Z_ratio = Z / Z_target
        
        voltage = self.compute_DMA_voltage(Z_target)
        
        # Peclet number
        Pe = np.abs(charge) * EL_CHARGE * voltage / (BOLTZMANN_CONSTANT * self.temperature) \
            * (1 - self.R1 / self.R2) / np.log(self.R2 / self.R1)
        
        sigma = np.sqrt(self.G * Z_ratio / Pe)
        
        TF = sigma / (np.sqrt(2) * self.beta * (1 - self.delta)) * (
            self._tf_eps((Z_ratio - (1 + self.beta)) / (np.sqrt(2) * sigma))
            + self._tf_eps((Z_ratio - (1 - self.beta)) / (np.sqrt(2) * sigma))
            - self._tf_eps((Z_ratio - (1 + self.beta * sigma)) / (np.sqrt(2) * sigma))
            - self._tf_eps((Z_ratio - (1 - self.beta * sigma)) / (np.sqrt(2) * sigma))
            )
    
        # Set possible negative values (numerical artifacts) to zero
        TF = np.clip(TF, 0, np.inf)
        
        return TF
    
    
    def compute_transfer_function(self):
        ''' Compute the DMA transfer function, i.e., the probability that a particle of mobility Z
        will be transmitted from the aerosol flow to the classified aerosol flow when classifying
        mobility Z_target.
        The transfer function is different for particles of different charges so it is calculated
        for each charge separately.
        '''
        
        # compute_using = 'bin centerpoint'
        compute_using = 'trapezoidal rule'
        # compute_using = 'Gaussian quadrature'
        
        transfer_function = np.zeros((
            self.charges.shape[0], self.d_m_data.shape[0], self.d_m.shape[0]
            ))
        
        binwidth = np.log10(self.d_m[1]) - np.log10(self.d_m[0])
        d_m_modelled_edges = np.logspace(np.log10(self.d_m[0]) - binwidth / 2,
                                np.log10(self.d_m[-1]) + binwidth / 2,
                                self.d_m.shape[0] + 1)
        
        if compute_using == 'trapezoidal rule':
            # Define the integration points for trapz
            n_intpts_per_bin = 3  # Including the start point but not the end point
            n_intpts = self.d_m.shape[0] * n_intpts_per_bin + 1
            # This includes the bin edges as well
            d_m_intpts = np.geomspace(d_m_modelled_edges[0], d_m_modelled_edges[-1], n_intpts)
            dx = np.log10(d_m_intpts[1]) - np.log10(d_m_intpts[0])
            weights = np.ones(n_intpts_per_bin + 1)
            weights[0] /= 2
            weights[-1] /= 2
        
        elif compute_using == 'Gaussian quadrature':
            # Define integration points for Gaussian quadrature
            n_bins = self.d_m.shape[0]
            log_d_m = np.log10(d_m_modelled_edges)
            a = log_d_m[:-1]
            b = log_d_m[1:]
            dx_sum = (a[0] + b[0]) / 2  # Assume the bin width stays the same for all bins
            dx_diff = (b[0] - a[0]) / 2
            # int_pts = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
            # weights = np.array([1, 1])
            # int_pts = np.array([-1 / np.sqrt(3 / 5), 0,  1 / np.sqrt(3 / 5)])
            # weights = np.array([5 / 9, 8 / 9, 5 / 9])
            int_pts = np.array([-np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
                                -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                                np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                                np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
                                ])
            weights = np.array([(18 - np.sqrt(30)) / 36, (18 + np.sqrt(30)) / 36,
                                (18 + np.sqrt(30)) / 36, (18 - np.sqrt(30)) / 36])
            d_m_gauss_pts = dx_diff * (int_pts + 2 * np.arange(n_bins)[:, np.newaxis]) + dx_sum
            d_m_gauss_pts = 10**d_m_gauss_pts
        
        for c_idx, charge in enumerate(self.charges):
            
            if compute_using == 'bin centerpoint':
                # Use the center-of-bin value for the transfer function
                # Mobilities of the particles we model
                Z_modelled = self.compute_electrical_mobility(self.d_m, charge)
                for z_idx, Z_target in enumerate(self.Z_targets):
                    
                    transfer_function[c_idx, z_idx] += (
                        # self._tf_element_nondiffusive(Z_modelled, Z_target)
                        self._tf_element_diffusive(Z_modelled, Z_target, charge)
                        ) #/ binwidth
            
            elif compute_using == 'trapezoidal rule':
                # Calculate the integral over the bin using the trapezoidal rule
                Z_modelled = self.compute_electrical_mobility(d_m_intpts, charge)
                for z_idx, Z_target in enumerate(self.Z_targets):
                    
                    tf_vals = self._tf_element_diffusive(Z_modelled, Z_target, charge)
                    
                    for d_m_idx, d_m_modelled in enumerate(self.d_m):
                        
                        start_idx = d_m_idx * n_intpts_per_bin
                        end_idx = (d_m_idx + 1) * n_intpts_per_bin + 1
                        
                        # tf_integral = 0.5 * np.sum(
                        #     (tf_vals[start_idx + 1 : end_idx] + tf_vals[start_idx : end_idx - 1])
                        #     * (np.log10(d_m_intpts[start_idx + 1 : end_idx])
                        #        - np.log10(d_m_intpts[start_idx : end_idx - 1])
                        #        )
                        #     ) / binwidth
                        tf_integral = np.sum(
                            (tf_vals[start_idx: end_idx] * weights) * dx
                            ) / binwidth
                        transfer_function[c_idx, z_idx, d_m_idx] += tf_integral
            
            
            elif compute_using == 'Gaussian quadrature':
                # Calculate the integral over the bin using Gaussian quadrature
                Z_modelled = self.compute_electrical_mobility(d_m_gauss_pts, charge)
                for z_idx, Z_target in enumerate(self.Z_targets):
                    
                    tf_vals = self._tf_element_diffusive(Z_modelled, Z_target, charge)
                    tf_integral = dx_diff * np.sum(tf_vals * weights, axis=1) / binwidth
                    transfer_function[c_idx, z_idx] += tf_integral
        
        return transfer_function
    
    
    def initialize_charging_model(self, model):
        ''' The charging model needs to be created (or loaded) at first.
        Here we also define some function wrappers to standardize the function call.
        The particle diameter input is optional; if it isn't given, the initialized
        self.d_m will be used. '''
        
        # Default values for the ion properties
        default_pos_mob = 1.35e-4
        default_neg_mob = 1.60e-4
        default_ratio = 1.0
        
        if model == 'Wiedensohler':
            
            self.charger = WiedensohlerChargingModel(self.d_m / 2, self.charges)
            
            def _charge_prob(self, pos_ion_mobility=None, neg_ion_mobility=None, ion_ratio=None):
                
                return self.charger.charging_probability()
            
        elif model == 'LYF-direct':
            
            self.charger = LYFChargingModel(self.d_m / 2, self.charges)
            
            def _charge_prob(self, 
                             pos_ion_mobility=default_pos_mob,
                             neg_ion_mobility=default_neg_mob,
                             ion_ratio=default_ratio,
                             ):
                
                return self.charger.charging_probability(pos_ion_mobility,
                                                         neg_ion_mobility,
                                                         ion_ratio
                                                         )
            
        elif model == 'LYF-interp':
            
            fname = resources.files('MPSS_UQ.data') / 'interpolator_flux_60dm_307'
            flux_interpolator = LYFFluxInterpolator(fname)
            
            self.charger = LYFChargingModel(self.d_m / 2,
                                            self.charges,
                                            flux_interpolator=flux_interpolator
                                            )
            
            def _charge_prob(self,
                             pos_ion_mobility=default_pos_mob,
                             neg_ion_mobility=default_neg_mob,
                             ion_ratio=default_ratio
                             ):
                
                return self.charger.charging_probability(pos_ion_mobility,
                                                         neg_ion_mobility,
                                                         ion_ratio
                                                         )
        
        else:
            raise Exception('Unknown charging model')
        
        self.charging_model = _charge_prob
        self.charging_model_name = model
    
    
    def compute_charging_probability(self,
                                     pos_ion_mobility=1.35e-4,
                                     neg_ion_mobility=1.60e-4,
                                     ion_ratio=1.0,
                                     ):
        
        return self.charging_model(self, pos_ion_mobility, neg_ion_mobility, ion_ratio)
    
    
    def compute_electrical_mobility(self, d_m, particle_charge):
        ''' Compute electrical mobility for spherical particles from the particle mobility
        diameter.
        particle_charge is given as the number of elementary charges on the particle.
        '''
        
        temperature = 273 + 20
        Kn = 2 * mean_free_path_air(temperature) / d_m  # Knudsen number
        eta = dynamic_viscosity(temperature)
        
        return np.abs(particle_charge) * EL_CHARGE * cunningham(Kn) / (3 * np.pi * eta * d_m)
    
    def compute_DMA_voltage(self, Z):
        ''' Compute the required voltages differences between the inner and outer DMA electrodes 
        to measure the desired mobilities.
        '''
        
        return (self.Qsh + self.Qe) / (4 * np.pi * self.length * Z) * np.log(self.R2 / self.R1)


class CondensationParticleCounter():
    
    def __init__(self, d_m, properties):
        
        self.d_m = d_m
        self.model = properties['CPC_model']
        self.output_type = properties['CPC_output_type']  # counts or concentration
        self.measuring_time = properties['CPC_measuring_time']  # Per one output bin (s)
        
        self.sample_flow = properties['Qa'] * lpm_to_m3s * 1e6
        self.sampled_volume = self.measuring_time * self.sample_flow
        
        self.counting_efficiency = self.calculate_counting_efficiency()
    
    
    def calculate_counting_efficiency(self):
        
        if self.model == 'Airmodus-A20':
            a = 0.96
            b = 6.86
            dp50 = 10.08
            
            count_efficiency = a * (1 - np.exp(np.log(2) * ((b - self.d_m * 1e9) / (dp50 - b))))
            return np.clip(count_efficiency, 0, 1)
        
        else:
            raise ValueError('Unknown CPC model.')
