# -*- coding: utf-8 -*-

import numpy as np
import importlib.resources as resources
import warnings

from scipy.special import erf
from functools import lru_cache

from MPSS_UQ.chargingmodels import (LYFChargingModel,
                                    LYFInterpolator,
                                    LYFFluxInterpolator,
                                    WiedensohlerChargingModel,
                                    ChargingModelWrapper,
                                    )
from MPSS_UQ.aerosol import (particle_diffusivity,
                             electrical_mobility,
                             BOLTZMANN_CONSTANT,
                             ELECTRON_CHARGE,
                             )


lpm_to_m3s = 1e-3 / 60  # Liters per minute to m3 per second conversion factor


class DifferentialMobilityParticleSizer:
    
    def __init__(self,
                 properties,             # A DMPS properties dictionary
                 inversion_grid='auto',  # Bins for inversion, "auto" or np.ndarray
                 n_bins=None,            # [Optional] If using inversion_grid=="auto", 
                                         # can choose the number of inversion bins
                 ):
        
        # Mobility diameters the device has been set to measure (corresponding to the
        # voltages measured)
        self.d_m_data = properties['d_m_data']
        
        # Diameters at which we want to calculate the solution of the inverse problem
        if isinstance(inversion_grid, str):
            if inversion_grid != 'auto':
                raise ValueError("inversion_grid must be 'auto' or a 1D numpy array.")
            self.d_m = _auto_inversion_grid(self.d_m_data, n_bins=n_bins)
        elif type(inversion_grid) == np.ndarray:
            self.d_m = inversion_grid
        else:
            raise ValueError("inversion_grid must be 'auto' or a 1D numpy array.")
        
        self.Qsh = properties['Qsh']  # Sheath flow (liters per minute)
        self.Qe = properties['Qe']  # Exhaust flow (lpm)
        self.Qa = properties['Qa']  # Aerosol flow (lpm)
        self.Qc = properties['Qc']  # Classified sample flow (lpm)
        # Convert flows from lpm to m^3/s
        self.Qsh *= lpm_to_m3s
        self.Qe *= lpm_to_m3s
        self.Qa *= lpm_to_m3s
        self.Qc *= lpm_to_m3s
        
        self.R1 = properties['R1']  # Inner radius of the DMA (m)
        self.R2 = properties['R2']  # Outer radius of the DMA (m)
        self.length = properties['L']  # Length of the DMA (m)
        self.L_eff = properties['L_eff']  # Effective DMA length for loss calculations (m)
        
        # Inlet sampling line        
        inlets = properties['inlets']
        if inlets is None:
            self.inlet_flows   = np.array([], dtype=float)
            self.inlet_lengths = np.array([], dtype=float)
        else:
            try:
                inlets = np.array(inlets, dtype=float)
                if inlets.ndim != 2 or inlets.shape[1] != 2:
                    raise ValueError("`inlets` must be a list of [flow, length] pairs.")
            except:
                raise ValueError("Invalid `inlets` format. Make sure the number of lengths and " + 
                                 "flow rates is the same.")
            self.inlet_lengths = inlets[:, 0]
            self.inlet_flows = inlets[:, 1]
        
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
        
        self.temperature = None  # Internal temperature (K)
        self.pressure = None  # Internal pressure (Pa)
        
        # Some derived values used later on
        self.compute_DMA_characteristic_values()
        
        # Particle charging model
        self.initialize_charging_model(properties['charging_model'])
        
        # Preallocate the system matrix
        self.system_matrix = np.zeros((self.d_m_data.shape[0], self.d_m.shape[0]))
        
        # Initialize the CPC
        self.CPC = CondensationParticleCounter(self.d_m_data, properties)
        
        self.operating_conditions_set = False
        
        if self.charging_model_name == 'Wiedensohler':
            # The Wiedensohler approximation requires no input parameters so we can
            # calculate it already here
            self.charging_probability = self.compute_charging_probability()
            self.charger_conditions_set = True
            
        else:
            self.charger_conditions_set = False
        
        self._calc_transfer_func_cacheable.cache_clear()
    
    
    def __str__(self):
        
        return (f"At {self.temperature - 273.15} °C and {self.pressure} Pa.\n" +
                f"Charging model: {self.charging_model_name}.\n" +
                f"CPC counting efficiency curve: {self.CPC.counting_efficiency_type}.\n"
                )
    
    
    def set_operating_conditions(self, temperature, pressure):
        ''' Set the operating temperature and pressure, then update all parts of the MPSS these
        values affect. Finally assemble the updated system matrix. '''
        
        # First quantize inputs (for LRU caching)
        
        # Round to nearest 2 K
        temperature_bin = int(round(temperature / 2.0) * 2)
        
        # Round to nearest 25 Pa
        pressure_bin = int(round(pressure / 25.0) * 25)
        
        self.temperature = temperature  # Internal temperature (K)
        self.pressure = pressure  # Internal pressure (Pa)
        
        self.transfer_function = self._calc_transfer_func_cacheable(temperature_bin, pressure_bin)
        
        self.penetration_efficiency = self.compute_penetration_efficiency()
        self.sampling_line_loss = self.compute_sampling_line_loss()
        
        # Only update the system matrix if the charging probability has been computed
        if self.charger_conditions_set:
            self._update_system_matrix()
        
        self.operating_conditions_set = True
    
    
    @lru_cache(maxsize=128)
    def _calc_transfer_func_cacheable(self, temperature, pressure):
        ''' A function to calculate the DMA transfer function in a cacheable way to use
        LRU caching for speedups when updating environmental conditions.
        Chech the cache stats with DMPS._calc_transfer_func_cacheable.cache_info()
        '''
        
        
        # Mobilities we want to classify
        Z_targets = electrical_mobility(self.d_m_data, temperature, pressure, 1)
        
        transfer_function = self.compute_transfer_function(Z_targets, temperature, pressure)
        
        return transfer_function
    
    
    def set_charger_properties(self, *args):
        ''' Calculate charging probability using input properties, and assemble
        the updated system matrix.
        Required inputs depend on the chosen charging model:
        
        LYF-interp:
            args = (positive_ion_mobility, negative_ion_mobility)
            
        LYF-direct or LYF-interp-flux:
            args = (positive_ion_mobility, negative_ion_mobility, ion_ratio)
        
        ion_ratio is the ratio of positive to negative ions, assumed equal to 1 in LYF-interp.
        '''
        
        if self.charging_model_name == 'Wiedensohler':
            raise Exception('Cannot change charger properties with the Wiedensohler model')
        
        self.charging_probability = self.compute_charging_probability(*args)
        
        # Only update the system matrix if the operating conditions have been set
        if self.operating_conditions_set:
            self._update_system_matrix()
        
        self.charger_conditions_set = True
    
    
    def forward_model(self, log10_N):
        ''' Run a sample with log size distribution log10_N through the measurement system. '''
        
        if not self.operating_conditions_set:
            raise RuntimeError("Operating conditions (temperature, pressure) must be set " +
                               "before running the forward model.")
        if not self.charger_conditions_set:
            raise RuntimeError("Charger properties must be set before running the forward model.")
        
        return self.system_matrix @ 10**log10_N
    
    
    def _update_system_matrix(self, charging_probability=None):
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
        self.system_matrix *= self.penetration_efficiency[:, np.newaxis]
        self.system_matrix *= self.sampling_line_loss[:, np.newaxis]
        
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
        
        D = particle_diffusivity(self.d_m_data, self.temperature, self.pressure)
        
        # Particle diffusion coefficient
        tau = np.pi * D * self.L_eff / self.Qa
        
        # Sherwood number for laminar flow
        Sh = a + b / (tau + c * tau**(1 / 3))
        
        return np.exp(-tau * Sh)
    
    
    def compute_sampling_line_loss(self):
        ''' Calculate sampling line losses. '''
        
        if self.inlet_lengths is None or len(self.inlet_lengths) == 0:
            return np.ones_like(self.d_m_data, dtype=float)
        if np.any(np.asarray(self.inlet_lengths) <= 0):
            raise ValueError("All inlet lengths must be > 0.")
        
        D = particle_diffusivity(self.d_m_data, self.temperature, self.pressure)
        
        mu = (self.inlet_lengths[:, None] * D[None, :]) / self.Qa
        
        P_small = 1.0 - 5.5 * np.power(mu, 2.0/3.0) + 3.77 * mu
        P_large = (0.819  * np.exp(-11.5 * mu)
                 + 0.0975 * np.exp(-70.1 * mu)
                 + 0.0325 * np.exp(-179  * mu))
        P = np.where(mu < 0.009, P_small, P_large)
    
        # Total penetration across all parts
        P_total = P.prod(axis=0)
        
        return P_total
    
    
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
    
    
    def _tf_element_diffusive(self, Z, Z_target, charge, temperature):
        ''' Compute a single element of the DMA transfer function with diffusion.
        Computed after Stolzenburg 1988 'An ultrafine aerosol size distribution system'.
        '''
        Z_ratio = Z / Z_target
        
        voltage = self.compute_DMA_voltage(Z_target)
        
        # Peclet number
        Pe = np.abs(charge) * ELECTRON_CHARGE * voltage / (BOLTZMANN_CONSTANT * temperature) \
            * (1 - self.R1 / self.R2) / np.log(self.R2 / self.R1)
        
        sigma = np.sqrt(self.G * Z_ratio / Pe)
        
        TF = sigma / (np.sqrt(2) * self.beta * (1 - self.delta)) * (
            self._tf_eps((Z_ratio - (1 + self.beta)) / (np.sqrt(2) * sigma))
            + self._tf_eps((Z_ratio - (1 - self.beta)) / (np.sqrt(2) * sigma))
            - self._tf_eps((Z_ratio - (1 + self.beta * self.delta)) / (np.sqrt(2) * sigma))
            - self._tf_eps((Z_ratio - (1 - self.beta * self.delta)) / (np.sqrt(2) * sigma))
            )
    
        # Set possible negative values (numerical artifacts) to zero
        TF = np.clip(TF, 0, np.inf)
        
        return TF
    
    
    def compute_transfer_function(self, Z_targets, temperature, pressure):
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
                Z_modelled = electrical_mobility(self.d_m, temperature, pressure, charge)
                for z_idx, Z_target in enumerate(Z_targets):
                    
                    transfer_function[c_idx, z_idx] += (
                        # self._tf_element_nondiffusive(Z_modelled, Z_target)
                        self._tf_element_diffusive(Z_modelled, Z_target, charge, temperature)
                        ) #/ binwidth
            
            elif compute_using == 'trapezoidal rule':
                # Calculate the integral over the bin using the trapezoidal rule
                Z_modelled = electrical_mobility(
                    d_m_intpts, temperature, pressure, charge)
                for z_idx, Z_target in enumerate(Z_targets):
                    
                    tf_vals = self._tf_element_diffusive(Z_modelled, Z_target, charge, temperature)
                    
                    for d_m_idx, d_m_modelled in enumerate(self.d_m):
                        
                        start_idx = d_m_idx * n_intpts_per_bin
                        end_idx = (d_m_idx + 1) * n_intpts_per_bin + 1
                        
                        tf_integral = np.sum(
                            (tf_vals[start_idx: end_idx] * weights) * dx
                            ) / binwidth
                        transfer_function[c_idx, z_idx, d_m_idx] += tf_integral
            
            
            elif compute_using == 'Gaussian quadrature':
                # Calculate the integral over the bin using Gaussian quadrature
                Z_modelled = electrical_mobility(
                    d_m_gauss_pts, temperature, pressure, charge)
                for z_idx, Z_target in enumerate(Z_targets):
                    
                    tf_vals = self._tf_element_diffusive(Z_modelled, Z_target, charge, temperature)
                    tf_integral = dx_diff * np.sum(tf_vals * weights, axis=1) / binwidth
                    transfer_function[c_idx, z_idx] += tf_integral
        
        return transfer_function
    
    
    def initialize_charging_model(self, model):
        ''' Initialize the charging model and assign a standardized callable. '''
    
        self.charging_model_name = model
    
        if model == 'Wiedensohler':
            self.charger = WiedensohlerChargingModel(self.d_m / 2, self.charges)
    
        elif model == 'LYF-direct':
            self.charger = LYFChargingModel(self.d_m / 2, self.charges)
    
        elif model == 'LYF-interp-flux':
            fname = resources.files('MPSS_UQ.data') / 'interpolator_flux_60dm_307'
            flux_interpolator = LYFFluxInterpolator(fname)
            self.charger = LYFChargingModel(
                self.d_m / 2, self.charges, flux_interpolator=flux_interpolator
                )
        
        elif model == 'LYF-interp':
            fname = resources.files('MPSS_UQ.data') / 'charging_prob_interpolator_data.npz'
            self.charger = LYFInterpolator(fname)
    
        else:
            raise Exception('Unknown charging model')
    
        self.charging_model = ChargingModelWrapper(model, self.charger, self.d_m, self.charges)
    
    
    def compute_charging_probability(self, *args):
        
        return self.charging_model(*args)
    
    
    def compute_DMA_voltage(self, Z):
        ''' Compute the required voltages differences between the inner and outer DMA electrodes 
        to measure the desired mobilities.
        '''
        
        return (self.Qsh + self.Qe) / (4 * np.pi * self.length * Z) * np.log(self.R2 / self.R1)


class CondensationParticleCounter:
    
    def __init__(self, d_m, properties):
        
        self.d_m = d_m
        self.output_type = properties['CPC_output_type']  # counts or concentration
        self.measuring_time = properties['CPC_measuring_time']  # Per one output bin (s)
        
        self.sample_flow = properties['Qa'] * lpm_to_m3s * 1e6
        self.sampled_volume = self.measuring_time * self.sample_flow
        
        # Choose which function to use to calculate the counting efficiency
        
        # returns None if key not found
        custom_function = properties.get('custom_CPC_count_eff_function')
        
        if custom_function:
            self.counting_efficiency = custom_function(self.d_m)
            self.counting_efficiency_type = 'custom'
        else:
            self.counting_efficiency = self.count_eff_ACTRIS_style(properties)
            self.counting_efficiency_type = 'ACTRIS style'
    
    
    def count_eff_ACTRIS_style(self, properties):
        
        a = properties['CPC_a']
        b = properties['CPC_b']
        dp50 = properties['CPC_dp50']
        
        if self.validate_input(a, b, dp50):
            count_efficiency = a * (1 - np.exp(np.log(2) * ((b - self.d_m * 1e9) / (dp50 - b))))
            
            return np.clip(count_efficiency, 0, 1)
    
    
    def validate_input(self, a, b, dp50):
        ''' Check the types and ranges of the input to the ACTRIS style counting efficiency
        calculation. '''

        # Type checks
        if not all(isinstance(x, (float)) for x in [a, b, dp50]):
            raise TypeError("All inputs must be float.")
    
        # Range checks
        if not (0 < a <= 1):
            raise ValueError("Input 'a' must be between 0 and 1.")
        if not (b >= 0):
            raise ValueError("Input 'b' must be non-negative.")
        if not (dp50 >= 0):
            raise ValueError("Input 'c' must be non-negative.")
    
        return True


def _auto_inversion_grid(d_m_data,
                         n_bins=None,
                         pad_decades=0.15,
                         min_d_m=1e-9,
                         max_d_m=2500e-9,
                         ):
    """
    Build a uniform log10-spaced inversion grid d_m from the measured size
    grid d_m_data by extending the range on both sides by a fixed number of
    decades. The inversion grid is clipped to [min_d_m, max_d_m].
    
    The upper bound for the inversion size range should be at least as high as the smallest
    of these two:
      1) Largest size bin for which the system (instrument) matrix is sensitive for.
         This should account for multiply charged particles, if modelled, such that if the
         largest size we set the instrument to measure is let's say 500 nm, we can still
         see in the largest channel doubly charged particles of size 1000 nm and 4 times
         charged particles of size 2000 nm. If we model up to 4 charges the instrument matrix
         is then sensitive to 2000 nm + half the transfer function width.
      2) Largest (singly charged) particles we expect to have in the measured aerosol.
         This could for example be the size above which an inertial impactor in the sampling
         inlet removes all particles.
    In practice, it is almost always safest to set the largest inversion bin size to the max
    size the charging interpolators work for (which is rather arbitrarily chosen to be 2500 nm
    at the moment). Sometimes the max size could be a bit smaller but it wouldn't bring much
    computational benefit.
    
    Parameters
    ----------
    d_m_data : (N,) array_like of float
        Measured bin centers (in meters). Assumed to be uniform in log10 spacing.
    n_bins : int or None, optional
        Total number of bins for the inversion grid. If `None` (default), the
        measured log10 step is reused; the grid is extended by at least one bin
        on each side and clipped to `[min_d_m, max_d_m]`.
    pad_decades : float, optional (default 0.15)
        Guard band in **decades** (log10) added below the measured range
        For example, 0.15 extends the range by about 1.41x (10**0.15 ≈ 1.41).
        If the range extends the absolute limits, it is clipped.
    min_d_m and max_d_m : float, optional
        Absolute minimum and maximum diameter the inversion grid is allowed to take.
        These are mainly related to the min and max diameters the charge fraction
        interpolators are configured for.

    Returns
    -------
    d_m : (M,) ndarray of float
        Strictly increasing inversion grid (in meters), uniform in log10 space.
        If `n_bins` is None, `M` is chosen so the measured log step is preserved
        over the extended/clipped range (with at least one extra bin per side).
        Otherwise `M == n_bins`.
    """
    
    # Calculate the inversion grid bounds
    d_min = max(min_d_m, d_m_data[0] / (10.0 ** pad_decades))
    d_max = max_d_m
    
    log_d_min = np.log10(d_min)
    log_d_max = np.log10(d_max)
    log_d_min_data = np.log10(d_m_data[0])
    log_d_max_data = np.log10(d_m_data[-1])
    
    # If we use d_m_data in the measured area, count the number of extra bins needed and
    # the new d_min and d_max in that 'grid'
    if n_bins is None:
        binwidth = np.log10(d_m_data[1]) - np.log10(d_m_data[0])
        
        # Make sure to extend at least one bin
        n_smaller_bins = max(1, int(np.floor((log_d_min_data - log_d_min) / binwidth)))
        n_larger_bins = max(1, int(np.floor((log_d_max - log_d_max_data) / binwidth)))
        log_d_min = log_d_min_data - n_smaller_bins * binwidth
        log_d_max = log_d_max_data + n_larger_bins * binwidth
        
        # But ensure that we don't go past the hard boundaries
        changed = False
        if 10**log_d_min < min_d_m:
            log_d_min = np.log10(min_d_m); changed = True
        if 10**log_d_max > max_d_m:
            log_d_max = np.log10(max_d_m); changed = True
        
        if changed:
            warnings.warn("Extended inversion grid was clipped by hard bounds; using the closest "
                          "allowable limit(s).", UserWarning)
        
        n_bins = int(np.round((log_d_max - log_d_min) / binwidth)) + 1

    # Recompute step to make endpoints exact, then build the grid
    log_grid = np.linspace(log_d_min, log_d_max, n_bins, dtype=float)
    
    d_m = 10**log_grid
    
    return d_m
