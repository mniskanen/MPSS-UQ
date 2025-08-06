# -*- coding: utf-8 -*-

import numpy as np
import dill as pickle

from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import quad
from scipy.special import hyp2f1
# from tqdm import tqdm
from dataclasses import dataclass


# import warnings
# warnings.filterwarnings("error")
# warnings.filterwarnings("default")


BOLTZMANN_CONSTANT = 1.380649e-23
ELECTRON_CHARGE = 1.60217663e-19
VACUUM_PERMITTIVITY = 8.8541878188e-12
AIR_RELATIVE_PERMITTIVITY = 1.00058986  # i.e., its dielectric constant


class LYFChargingModel():
    ''' Implement the Lopez-Yglesias & Flagan charging model.
    Input:
        particle_radius : radius of the aerosol particles of interest
        particle_charges_output : sign and number of elementary charges on the particles
        temperature : temperature
        max_considered_charge : how many charges are taken into account when calculating the 
                                charge fractions
        flux_interpolator : (optional) an interpolator object (pre-computed) that replaces
                            the expensive calculations of the average flux coefficients by
                            a quick interpolation. This is the recommended way to use this class
                            for computing the charging probabilities in practice.
    '''
    
    def __init__(self,
                 particle_radius,
                 particle_charges_output,
                 temperature=None,
                 max_considered_charge=None,
                 flux_interpolator=None
                 ):
        
        self.particle_radius = particle_radius
        
        # The charges whose probabilities will be returned
        self.particle_charges_out = particle_charges_output
        
        if flux_interpolator is not None:
            self.use_flux_interpolator = True
            self.flux_interpolator = flux_interpolator
        else:
            self.use_flux_interpolator = False
        
        # Calculate (internally) up to this number of charges for the LYF model to be accurate
        if max_considered_charge:
            if self.use_flux_interpolator:
                raise Warning('Flux interpolator overrides manual max_considered_charge')
            
            assert max_considered_charge >= np.max(np.abs(self.particle_charges_out))
            max_charge = max_considered_charge
            
        else:
            max_charge = 25
        
        if self.use_flux_interpolator:
            max_charge = flux_interpolator.max_considered_charge
        
        self.particle_charges = np.arange(-max_charge, max_charge + 1)
        
        if temperature:
            assert temperature > 270 and temperature < 370
            self.temperature = temperature
        else:
            self.temperature = 298.15  # Kelvins
        
        self.pressure = 101325  # Pa
        self.ion_charges = np.array([-1, 1], dtype=np.int32)
        
        self.gas_mass = 28.97 * 1.660539040e-27  # kg (from dalton (Da))
        self.gas_viscosity = 18.27e-6  # Pa * s
        # This can be computed here because it depends only on temperature, gas mass and viscosity
        self.gas_effective_radius = self.gas_molecule_effective_radius()
        
        self.average_flux_coefficients = None
        
        # Preset initial values
        pos_ion_mobility = 1.2e-4  # m^2 V^-1 s^-1
        neg_ion_mobility = 1.35e-4  # m^2 V^-1 s^-1
        self.update_ion_parameters(pos_ion_mobility, neg_ion_mobility)
        
        self.show_progressbar = True
    
    
    def update_ion_parameters(self, pos_ion_mobility, neg_ion_mobility):
        
        # Update the masses based on mobilities
        pos_ion_mass = self.mobility_to_mass(pos_ion_mobility)
        neg_ion_mass = self.mobility_to_mass(neg_ion_mobility)
        
        # Or use these set values
        # pos_ion_mass = 150 * 1.660539040e-27  # kg (from dalton (Da))
        # neg_ion_mass = 90 * 1.660539040e-27  # kg (from dalton (Da))
        
        self.positive_ion = self.calculate_ion_properties(1, pos_ion_mass, pos_ion_mobility)
        self.negative_ion = self.calculate_ion_properties(-1, neg_ion_mass, neg_ion_mobility)
    
    
    def mobility_to_mass(self, mobility):
        
        model = 'Kilpatrick'
        # model = 'own fit'
        
        if model == 'Kilpatrick':
            # Solve Kilpatrick's model for mobility to mass
            a = -0.0347
            b = -0.0376
            c = 1.4662
            mass = np.exp((-b - np.sqrt(b**2 - 4 * a * (c - np.log(mobility * 1e4)))) / (2 * a))
            
            # Convert to kg
            mass *= 1.660539040e-27
        
        elif model == 'own fit':
            # Coefficients on an exponential model based on a LS fit to some data points
            coeffs = np.array([6.56664140e+00, -1.20354477e+04])
            
            # Compute mass and convert to kg
            mass = np.exp(coeffs[0]) * np.exp(coeffs[1] * mobility) * 1.660539040e-27
        
        return mass
    
    
    def charging_probability(self,
                             pos_ion_mobility,
                             neg_ion_mobility,
                             ion_ratio=1.0,
                             ):
        ''' Compute the charging probaility based on the flux coefficients.

        pos_ion_mobility = mobility of the positive ion
        neg_ion_mobility = mobility of the negative ion
        ion_ratio = ratio of positive to negative charging ions
        
        '''
        
        # if average_flux_coefficients is None:
        #     average_flux_coefficients = self.compute_average_flux_coefficients()
        
        if self.use_flux_interpolator:
            self.average_flux_coefficients = self.flux_interpolator(2 * self.particle_radius,
                                                                    pos_ion_mobility,
                                                                    neg_ion_mobility,
                                                                    )
        
        else:
            self.update_ion_parameters(pos_ion_mobility, neg_ion_mobility)
            self.average_flux_coefficients = self.compute_average_flux_coefficients()
        
        # Compute charge state ratios N_k / N_(k-1)
        
        non_neutral_charges = np.delete(self.particle_charges, self.particle_charges == 0)
        
        # Exclude zero
        charge_state_successive_ratios = np.zeros((non_neutral_charges.shape[0],
                                                   self.particle_radius.shape[0]))
        charge_state_to_neutral_ratios = np.zeros_like(charge_state_successive_ratios)
        
        # Compute the successive ratios
        for k_idx, k in enumerate(self.particle_charges):
            if k < 0:
                charge_state_successive_ratios[k_idx] = (
                    self.average_flux_coefficients[0,  k_idx + 1]
                    / self.average_flux_coefficients[1, k_idx]
                    ) / ion_ratio
            elif k > 0:
                # "decrease" k_idx by one here (because skipped the 0 charge):
                charge_state_successive_ratios[k_idx - 1] = (
                    self.average_flux_coefficients[1, k_idx - 1]
                    / self.average_flux_coefficients[0, k_idx]
                    ) * ion_ratio
            else:
                pass
        
        # Compute ratios of charge state k to neutral state
        # Loop over all particle charge states excluding the neutral charge
        for k_idx, k in enumerate(non_neutral_charges):
            if k < 0:
                charge_state_to_neutral_ratios[k_idx] = np.prod(
                    charge_state_successive_ratios[k_idx : k_idx + np.abs(k)], axis=0
                    )
            elif k > 0:
                charge_state_to_neutral_ratios[k_idx] = np.prod(
                    charge_state_successive_ratios[k_idx - np.abs(k) + 1 : k_idx + 1] , axis=0
                    )
        
        total_charged_to_neutral_ratio = np.sum(charge_state_to_neutral_ratios, axis=0) + 1.
        
        charge_state_probability = (
            charge_state_to_neutral_ratios / total_charged_to_neutral_ratio[np.newaxis, :]
            )
        
        # Add probability for neutral state (everything has to sum to one)
        neutral_probability = (np.ones((1, self.particle_radius.shape[0]))
                               - np.sum(charge_state_probability, axis=0, keepdims=True)
                              )
        
        n_neg = np.sum(self.particle_charges < 0)
        charge_state_probability = np.r_[
            charge_state_probability[:n_neg],
            neutral_probability,
            charge_state_probability[n_neg:]
            ]
        
        charge_state_probability = np.clip(charge_state_probability, 1e-16, 1)
        
        # Return only what was requested by the user
        return_idx = np.nonzero(self.particle_charges[:, None] == self.particle_charges_out)[0]
        
        return charge_state_probability[return_idx]
    
    
    def compute_average_flux_coefficients(self):
        ''' Compute the ion attachment (flux) coefficient, \beta, averaged over the Maxwell
        speed distribution. '''
        
        # Preallocate
        avg_flux_coefficients = np.zeros((self.ion_charges.shape[0],
                                          self.particle_charges.shape[0],
                                          self.particle_radius.shape[0]))
        
        def integrand(v, ion, r_0, r_contact, r_p, k):
            return (self.flux_coefficient(v, r_0, r_contact, r_p, k, ion)
                    * self.Maxwell_probability(v, ion.mass, self.temperature))
        
        if self.show_progressbar:
            pbar = tqdm(position=0, desc='Computing flux coefficients')
            total_iterations = 2 * self.particle_charges.shape[0] * self.particle_radius.shape[0]
            pbar.reset(total = total_iterations)
        for i_idx, i in enumerate(self.ion_charges):
            
            # Things needed to compute ion effective radius
            if i > 0:
                ion = self.positive_ion
            elif i < 0:
                ion = self.negative_ion
            
            for k_idx, k in enumerate(self.particle_charges):
                for r_idx, r_p in enumerate(self.particle_radius):
                    
                    # Radius at which the ion would make physical contact
                    r_contact = ion.effective_radius + r_p
                    # Radius of the limiting sphere
                    r_0 = r_contact + ion.mean_free_path
                    
                    # Numerical integration using quad from scipy
                    # The upper limit of 1000 m/s seems to be enough for accurate results
                    avg_flux_coefficients[i_idx, k_idx, r_idx], _ = quad(
                        integrand, 0, 1000, args=(ion, r_0, r_contact, r_p, k)
                        )
                    
                    if self.show_progressbar:
                        # Manually update progress bar
                        pbar.update(1)
        
        if self.show_progressbar:
            pbar.refresh()
        
        return avg_flux_coefficients
    
    
    def flux_coefficient(self, c_0, r_0, r_contact, r_p, k, ion):
        ''' Compute the flux coefficient for c_0, i.e. the ion speed at the radius of the limiting
        sphere r_0, k, and i (charges of the particle and ion, respectively).
        
        r_p = radius of the particle
        
        '''
        
        # assert type(i) == np.int32
        
        r_a, b_0_squared = self.capture_cross_section(k, ion, c_0, r_contact, r_0, r_p)
        
        delta = self.three_body_trapping_radius(ion, c_0, k, r_0, r_p, r_a)
                
        if r_a > delta:
            f = 1
        else:
            b_delta_squared = self.b_squared(delta, r_0, r_p, k, ion, c_0)
            f = self.ion_capture_probability(delta, c_0, ion, b_0_squared, b_delta_squared)
        
            assert b_delta_squared >= b_0_squared
        
        # Use simple trapezoidal integration, but space the integation bins logarithmically
        # (the integrand seems to behave quite smoothly on a log-scale and spans maaaaaany
        # orders of magnitude, so something like scipy's quad is not the best suited to evaluate
        # it). Also, the integrand changes fastest in the beginning of the range, so let's  make
        # the bins denser there.
        n_r_dense = 40
        n_r_coarse = 50
        r_vec = np.zeros(n_r_dense + n_r_coarse)
        r_vec[:n_r_dense] = np.geomspace(r_0, 1e2 * r_0, num=n_r_dense, endpoint=False)
        r_vec[n_r_dense:] = np.geomspace(1e2 * r_0, 1e12 * r_0, num=n_r_coarse)

        integrand = np.zeros(r_vec.shape[0])
        for idx, r in enumerate(r_vec):
            integrand[idx] = r**-2 * np.exp(self.potential_energy(k, ion, r, r_p) / (
                BOLTZMANN_CONSTANT * self.temperature))
        
        integral = 0.5 * np.sum((integrand[:-1] + integrand[1:]) * (r_vec[1:] - r_vec[:-1]))
        
        common_term = np.pi * c_0 * b_0_squared * f * np.exp(
            -self.potential_energy(k, ion, r_0, r_p) / (
                BOLTZMANN_CONSTANT * self.temperature)
            )
        
        beta = common_term / (
            1 + common_term / (4 * np.pi * ion.diffusivity) * integral
            )
        
        return beta
    
    
    def ion_capture_probability(self, delta, c_0, ion, b_0_squared,
                                b_delta_squared):
        ''' Compute the probability that the ion will be captured by the particle, accounting for
        three-body trapping. '''
        
        argument = np.sqrt(b_0_squared / b_delta_squared)
        argument = np.min((1, argument))
        argument = np.max((-1, argument))
        theta_c = np.arcsin(argument)
        
        f = 1 - ion.mean_free_path**2 / (2 * delta**2) * (
            1 - np.exp(-2 * delta * np.cos(theta_c) / ion.mean_free_path)
            * (1 + 2 * delta * np.cos(theta_c) / ion.mean_free_path)
            )
        
        return f
    
    
    def three_body_trapping_radius(self, ion, c_0, k, r_0, r_p, r_a):
        
        def functional(r, gas_mass, ion, c_0, k, r_0, r_p, temperature):
            ''' When this functional returns zero (as a function of r), we have found r = delta.
            '''
            
            mass_term = (ion.mass**2 + gas_mass**2) / (ion.mass + gas_mass)**2
            
            # (Pre-)Compute potential energies that we need to use multiple times
            potential_r0_rp = self.potential_energy(k, ion, r_0, r_p)
            potential_r_rp = self.potential_energy(k, ion, r, r_p)
            
            # Mean speed (c_f) of ion after collision with gas molecule
            mean_speed = (
                2 / gas_mass * 3 * BOLTZMANN_CONSTANT * temperature / 2 * (mass_term + 1)
                + 2 / ion.mass
                * (ion.mass / 2 * c_0**2 + potential_r0_rp - potential_r_rp) * mass_term
                )
            
            return potential_r0_rp - potential_r_rp - ion.mass / 2 * mean_speed
        
        # Find the root of the above functional to find the three-body trapping radius \delta
        a = r_a
        b = r_0
        fa = functional(
            a, self.gas_mass, ion, c_0, k, r_0, r_p, self.temperature)
        fb = functional(
            b, self.gas_mass, ion, c_0, k, r_0, r_p, self.temperature)
        
        if fa * fb > 0:
            # This (probably...) means \delta will be smaller than r_a, i.e. the ion will be
            # caught regardless and three-body trapping doesn't play a part.
            delta = 0.9 * r_a  # Something (anything) smaller than r_a
            return delta
        
        # Find the root using Brent's method
        delta = brentq(functional, a, b,
                           args=(self.gas_mass, ion, c_0, k, r_0, r_p, self.temperature)
                       )
        
        return delta
    
    
    def b_squared(self, r, r_0, r_p, k, ion, c_0):
        
        b2 = r**2 * (1 - 2 * (self.potential_energy(k, ion, r, r_p)
                                  - self.potential_energy(k, ion, r_0, r_p))
                          / (ion.mass * c_0**2))
        
        return b2
    
    
    def capture_cross_section(self, k, ion, c_0, r_contact, r_0, r_p):
        ''' Compute values of
        b_0 (radius of the modified capture cross-section) and
        r_a (radius of the interaction cross-section). '''
        
        options = {'xatol' : 1e-12}
        res = minimize_scalar(self.b_squared,
                              bounds=(r_contact, r_0),
                              args=(r_0, r_p, k, ion, c_0),
                              method='bounded',
                              options=options
                              )
        
        r_a = res.x
        b_0_squared = res.fun
        if b_0_squared < 0:
            b_0_squared = 0
        
        return r_a, b_0_squared
    
    
    def gas_molecule_effective_radius(self):
        gas_rad = (self.gas_mass * self.temperature * BOLTZMANN_CONSTANT / (
            16 * np.pi**3 * self.gas_viscosity**2))**(1 / 4)
        
        return gas_rad
    
    
    @dataclass
    class Ion:
        ''' Class for storing ion properties. '''
        
        charge: int
        mobility: float
        mass: float
        effective_radius: float
        mean_free_path: float
        diffusivity: float
    
    
    def calculate_ion_properties(self, charge, ion_mass, ion_mobility):
        ''' Compute the effective radius, mean free path, and diffusivity of an ion based on its
        mass and mobility. '''
        
        ion_mean_speed = np.sqrt(8 * BOLTZMANN_CONSTANT * self.temperature / (
            np.pi * ion_mass))
        
        ion_diffusivity = ion_mobility * BOLTZMANN_CONSTANT * self.temperature / ELECTRON_CHARGE
        
        ion_mean_free_path = 32 * ion_diffusivity \
            / (3 * np.pi * (1 + ion_mass / self.gas_mass) * ion_mean_speed)
        
        ion_effective_radius = -self.gas_effective_radius + (
            3 * (1 + ion_mass / self.gas_mass)**(1 / 2) * ion_mean_speed
            * BOLTZMANN_CONSTANT * self.temperature
            / (8 * self.pressure * ion_diffusivity)
            )**(1/2)
        
        assert ion_mean_free_path > 0
        assert ion_effective_radius > 0
        
        return self.Ion(charge, ion_mobility, ion_mass, ion_effective_radius,
                        ion_mean_free_path, ion_diffusivity)
    
    
    def Maxwell_probability(self, v, mass, temperature):
        ''' The Maxwell probability distribution of particle speeds. '''
        
        return ((mass / (2 * np.pi * BOLTZMANN_CONSTANT * temperature))**(3 / 2)
             * 4 * np.pi * v**2 * np.exp(
                 -mass * v**2 / (2 * BOLTZMANN_CONSTANT * temperature)))
    
    
    def potential_energy(self, k, ion, r, r_p):
        # Potential energy between a particle of charge k and an ion of charge i at a distance r.
        
        chi_p = 80  # Dielectric constant of the particle (water)
        # chi_p = 2.6  # Dielectric constant of the particle (polystyrene)
        chi_i = 100 # Dielectric constant of the ion
        
        gamma_p = (chi_p - AIR_RELATIVE_PERMITTIVITY) / (chi_p + AIR_RELATIVE_PERMITTIVITY)
        gamma_i = (chi_i - AIR_RELATIVE_PERMITTIVITY) / (chi_i + AIR_RELATIVE_PERMITTIVITY)
        
        xi_p = r_p**2 / r**2
        xi_i = ion.effective_radius**2 / r**2
        
        # Compute the integrals (semi-)analytically. I have tested this against trapezoidal
        # integration, and it seems to converge to this as the number of bins grows.
        
        a_p = (3 - gamma_p) / 2
        # There is sometimes an issue with the hypergeometric funcion (it evaluates to inf),
        # namely when its z-value (here xi_p) is close to one and the other inputs are
        # "just right". In that case, find a z-value (the upper integration limit) that can
        # be evaluated and compute the rest with trapezoidal integration.
        upper_limit = 1
        hyper_func = hyp2f1(2, a_p, a_p + 1, xi_p * upper_limit)
        while np.isinf(hyper_func):
            upper_limit *= 0.999  # Decrease the upper integration limit
            hyper_func = hyp2f1(2, a_p, a_p + 1, xi_p * upper_limit)
        
        integral_p_part1 = 1 / (2 * a_p) * upper_limit**a_p * hyper_func
        
        integral_p_part2 = 0
        if upper_limit < 1:
            v_vec = np.linspace(upper_limit, 1, 50)
            integrand = v_vec**((1 - gamma_p) / 2) / (2 * (1 - v_vec * xi_p)**2)
            integral_p_part2 = 0.5 * np.sum(
                (integrand[:-1] + integrand[1:]) * (v_vec[1:] - v_vec[:-1])
                )
        
        integral_p = integral_p_part1 + integral_p_part2
        
        # Didn't encounter the same issues with this integral so go just like this
        a_i = (3 - gamma_i) / 2
        integral_i = 1 / (2 * a_i) * hyp2f1(2, a_i, a_i + 1, xi_i)
        
        phi = ELECTRON_CHARGE**2 / (
            4 * np.pi * AIR_RELATIVE_PERMITTIVITY * VACUUM_PERMITTIVITY * r
            ) * (ion.charge * k - gamma_p * xi_p**(3 / 2) * integral_p
                 -k**2 * gamma_i * xi_i**(3 / 2) * integral_i)
        
        return phi


class LYFInterpolator():
    ''' Fast evaluation of the LYF model charging probability.
    Based on (pre)computing a representative grid of values and an interpolator object for them.
    Evaluation consists then only of evaluating the interpolator.
    
    LYF model parameters currently implemented:
        - mobility of positive ions
        - mobility of negative ions
    '''
    
    def __init__(self, saved_interpolator_file=None):
        ''' This class is used for both precomputing and saving the interpolator to file,
        as well as loading it from file and evaluating. Therefore most of the initializations are
        done when the use case is known (save / load). '''
        
        if saved_interpolator_file:
            self.load_interpolators(saved_interpolator_file)
            
            # Get the interpolator bounds (assume each interpolator has the same bounds)
            self.d_m_bounds = np.array([10**self.interpolator[0].grid[0][0],
                                        10**self.interpolator[0].grid[0][-1]])
            self.pos_ion_mobility_bounds = 1e-4 * np.array([10**self.interpolator[0].grid[1][0],
                                                            10**self.interpolator[0].grid[1][-1]])
            self.neg_ion_mobility_bounds = 1e-4 * np.array([10**self.interpolator[0].grid[2][0],
                                                            10**self.interpolator[0].grid[2][-1]])
            
            # The charges modelled # TODO: make this a part of the saved file
            self.charges_output = np.arange(-8, 8 + 1)
            
        else:
            self.interpolator = None
    
    
    def construct_interpolators(self, charges_output=None, parameter_bounds=None):
        
        if charges_output is None:
            self.charges_output = np.arange(-8, 8+1)
        else:
            self.charges_output = charges_output
        
        if parameter_bounds is None:
            parameter_bounds = {
                'min_d_m' : 1e-9,  # m
                'max_d_m' : 2500e-9,
                'min_pos_ion_mobility' : 1.05e-4,  # m^2 V^-1 s^-1
                'max_pos_ion_mobility' : 1.70e-4,
                'min_neg_ion_mobility' : 1.05e-4,
                'max_neg_ion_mobility' : 2.10e-4,
                }
        
        self.n_charges = self.charges_output.shape[0]
        
        self.d_m_bounds = np.array([parameter_bounds['min_d_m'],
                                    parameter_bounds['max_d_m']])
        self.pos_ion_mobility_bounds = np.array([parameter_bounds['min_pos_ion_mobility'],
                                                 parameter_bounds['max_pos_ion_mobility']])
        self.neg_ion_mobility_bounds = np.array([parameter_bounds['min_neg_ion_mobility'],
                                                 parameter_bounds['max_neg_ion_mobility']])
        
        self.n_d_m = 30
        self.n_mob = 10
        self.d_m = np.geomspace(self.d_m_bounds[0], self.d_m_bounds[1], self.n_d_m)
        self.pos_ion_mobility = np.linspace(self.pos_ion_mobility_bounds[0],
                                            self.pos_ion_mobility_bounds[1], self.n_mob)
        self.neg_ion_mobility = np.linspace(self.neg_ion_mobility_bounds[0],
                                            self.neg_ion_mobility_bounds[1],self.n_mob)
        
        self.charging_model = LYFChargingModel(self.d_m / 2, self.charges_output,
                                               max_considered_charge=25)
        
        # Multiple tqdm progress bars don't work with some IDEs, so disable the charging model one
        self.charging_model.show_progressbar = False
        
        # Construct interpolators -----------
        # Preallocate
        values = np.zeros((self.n_charges, self.n_d_m, self.n_mob, self.n_mob))
        
        # One interpolator per charge
        self.interpolator = [ [] for _ in range(self.n_charges)]
        
        pbar = tqdm(position=0, desc='Creating interpolators')
        pbar.reset(total = self.n_mob * self.n_mob)
        for i, pos_ion_mob in enumerate(self.pos_ion_mobility):
            for j, neg_ion_mob in enumerate(self.neg_ion_mobility):
                
                # Negative ion mobility should be greater than that of the positive ion, but
                # for the interpolator here leave a 5 % margin for numerical reasons
                # if neg_ion_mob > 0.95 * pos_ion_mob:
                breakpoint()
                self.charging_model.update_ion_parameters(pos_ion_mob, neg_ion_mob)
                values[:, :, i, j] = self.charging_model.charging_probability()
                # else:
                    # These values will never be interpolated so no need to run the model,
                    # just set to something practically zero
                    # values[:, i, j, :] = 1e-16 * np.ones_like(values[:, i, j, :])
                
                # # Set the minimum value to something that we can still take log of
                # values[:, i, j, :] = np.clip(values[:, i, j, :], 1e-30, np.inf)
                
                # if np.any(values[:, i, j, :] == 0):
                #     breakpoint()
                
                pbar.update(1)
        
        # Construct the interpolator objects, separately for each charge
        # For numerical reasons, scale all inputs to approximately [-10, 10]
        log_diam = np.log10(self.d_m)
        scaled_pos_mob = self.pos_ion_mobility * 1e4
        scaled_neg_mob = self.neg_ion_mobility * 1e4
        
        for i in range(self.n_charges):
            try:
                self.interpolator[i] = RegularGridInterpolator(
                    (log_diam, scaled_pos_mob, scaled_neg_mob), np.log10(values[i]),
                    method='cubic'
                    )
            except:
                breakpoint()
        
        pbar.refresh()
    
    
    def __call__(self, d_m, pos_mobility, neg_mobility, charges=None):
        ''' Evaluate the charging probabilities for the given input parameters.
        The input d_m can be a vector.
        '''
        # TODO: could implement here a check for the input being within the interpolator bounds
        
        if charges is None:
            # Return all charges
            n_charges = len(self.interpolator)
            charge_idx = np.arange(n_charges)
            
        else:
             # We're requesting only some of the available charges
             charge_idx = np.nonzero(self.charges_output[:, None] == charges)[0]
             n_charges = charges.shape[0]
        
        # scale
        log_diam = np.log10(d_m)
        scaled_pos_mob = pos_mobility * 1e4
        scaled_neg_mob = neg_mobility * 1e4
        
        charge_probabilities = np.zeros((n_charges, d_m.shape[0]))
        for count, chrg_idx in enumerate(charge_idx):
            charge_probabilities[count] = self.interpolator[chrg_idx](
                (log_diam, scaled_pos_mob, scaled_neg_mob)
                )
        
        return 10**charge_probabilities  # Scale back
    
    
    def save_interpolators(self, fname):
        with open(fname, 'wb') as outp:
            pickle.dump(self.interpolator, outp, pickle.HIGHEST_PROTOCOL)
    
    
    def load_interpolators(self, fname):
        with open(fname, 'rb') as inp:
            self.interpolator = pickle.load(inp)


class LYFFluxInterpolator():
    ''' Fast evaluation of the LYF model average flux coefficients.
    Based on (pre)computing a representative grid of values and an interpolator object for them.
    Evaluation consists then only of evaluating the interpolator.
    
    LYF model parameters currently implemented:
        - mobility of positive ions
        - mobility of negative ions
    '''
    
    def __init__(self, saved_interpolator_file=None):
        ''' This class is used for both precomputing and saving the interpolator to file,
        as well as loading it from file and evaluating. Therefore most of the initializations are
        done when the use case is known (save / load). '''
        
        if saved_interpolator_file:
            self.load_interpolators(saved_interpolator_file)
            
            # Get the interpolator bounds (assume each interpolator has the same bounds)
            self.d_m_bounds = np.array([10**self.interpolator[0][0].grid[0][0],
                                        10**self.interpolator[0][0].grid[0][-1]])
            self.pos_ion_mobility_bounds = 1e-4 * np.array([10**self.interpolator[0][0].grid[1][0],
                                                            10**self.interpolator[0][0].grid[1][-1]
                                                            ])
            self.neg_ion_mobility_bounds = 1e-4 * np.array([10**self.interpolator[0][0].grid[2][0],
                                                            10**self.interpolator[0][0].grid[2][-1]
                                                            ])
            
            # A hacky way to get the max modelled charge
            self.n_charges = len(self.interpolator[0])
            self.max_considered_charge = int((self.n_charges - 1) / 2)
        
        else:
            self.interpolator = None
    
    
    def construct_interpolators(self, parameter_bounds=None):
        
        max_considered_charge = 25
        
        self.charges_output = np.arange(-max_considered_charge, max_considered_charge+1)
        
        if parameter_bounds is None:
            parameter_bounds = {
                'min_d_m' : 1e-9,  # m
                'max_d_m' : 2500e-9,
                'min_pos_ion_mobility' : 1.05e-4,  # m^2 V^-1 s^-1
                'max_pos_ion_mobility' : 1.70e-4,
                'min_neg_ion_mobility' : 1.05e-4,
                'max_neg_ion_mobility' : 2.10e-4,
                # 'min_pos_ion_mobility' : 1.05e-4,  # m^2 V^-1 s^-1
                # 'max_pos_ion_mobility' : 1.70e-4,
                # 'min_neg_ion_mobility' : 1.05e-4,
                # 'max_neg_ion_mobility' : 2.10e-4,
                }
        
        self.n_charges = self.charges_output.shape[0]
        
        self.d_m_bounds = np.array([parameter_bounds['min_d_m'],
                                    parameter_bounds['max_d_m']])
        self.pos_ion_mobility_bounds = np.array([parameter_bounds['min_pos_ion_mobility'],
                                                 parameter_bounds['max_pos_ion_mobility']])
        self.neg_ion_mobility_bounds = np.array([parameter_bounds['min_neg_ion_mobility'],
                                                 parameter_bounds['max_neg_ion_mobility']])
        
        self.n_d_m = 60  # 30 # testing if this helps the "Gibbs" phenomenon
        self.n_mob = 10
        self.d_m = np.geomspace(self.d_m_bounds[0], self.d_m_bounds[1], self.n_d_m)
        self.pos_ion_mobility = np.linspace(self.pos_ion_mobility_bounds[0],
                                            self.pos_ion_mobility_bounds[1], self.n_mob)
        self.neg_ion_mobility = np.linspace(self.neg_ion_mobility_bounds[0],
                                            self.neg_ion_mobility_bounds[1],self.n_mob)
        
        self.charging_model = LYFChargingModel(self.d_m / 2, self.charges_output,
                                               max_considered_charge=max_considered_charge)
        
        # Multiple tqdm progress bars don't work with some IDEs, so disable the charging model one
        self.charging_model.show_progressbar = False
        
        # Construct interpolators -----------
        # Preallocate
        # (nmbr of ion charges (2), nmbr of charges, nmbr of particle sizes, nmbrs of mobilities)
        flux_coeffs = np.zeros((2, self.n_charges, self.n_d_m, self.n_mob, self.n_mob))
        # values = np.zeros((self.n_charges, self.n_d_m, self.n_mob, self.n_mob))
        
        # One interpolator per charge and ion sign
        self.interpolator_neg_ion = [ [] for _ in range(self.n_charges)]
        self.interpolator_pos_ion = [ [] for _ in range(self.n_charges)]
        
        pbar = tqdm(position=0, desc='Creating interpolators')
        pbar.reset(total = self.n_mob * self.n_mob)
        for i, pos_ion_mob in enumerate(self.pos_ion_mobility):
            for j, neg_ion_mob in enumerate(self.neg_ion_mobility):
                
                # Negative ion mobility should be greater than that of the positive ion, but
                # for the interpolator here leave a 5 % margin for numerical reasons
                # if neg_ion_mob > 0.95 * pos_ion_mob:
                self.charging_model.update_ion_parameters(pos_ion_mob, neg_ion_mob)
                flux_coeffs[:, :, :, i, j] = self.charging_model.compute_average_flux_coefficients()
                # else:
                    # These values will never be interpolated so no need to run the model,
                    # just set to something practically zero
                    # values[:, i, j, :] = 1e-16 * np.ones_like(values[:, i, j, :])
                
                # # Set the minimum value to something that we can still take log of
                # values[:, i, j, :] = np.clip(values[:, i, j, :], 1e-30, np.inf)
                
                # if np.any(values[:, i, j, :] == 0):
                #     breakpoint()
                
                pbar.update(1)
        
        # Construct the interpolator objects, separately for each charge
        # For numerical reasons, scale all inputs to approximately [-10, 10]
        log_diam = np.log10(self.d_m)
        scaled_pos_mob = self.pos_ion_mobility * 1e4
        scaled_neg_mob = self.neg_ion_mobility * 1e4
        
        for i in range(self.n_charges):
            try:
                self.interpolator_neg_ion[i] = RegularGridInterpolator(
                    (log_diam, scaled_pos_mob, scaled_neg_mob),
                    # np.log10(np.clip(flux_coeffs[0, i], 1e-16, np.inf)),
                    flux_coeffs[0, i],
                    # method='cubic'
                    )
                self.interpolator_pos_ion[i] = RegularGridInterpolator(
                    (log_diam, scaled_pos_mob, scaled_neg_mob),
                    # np.log10(np.clip(flux_coeffs[1, i], 1e-16, np.inf)),
                    flux_coeffs[1, i],
                    # method='cubic'
                    )
            except:
                breakpoint()
        
        pbar.refresh()
        
        self.interpolator = (self.interpolator_neg_ion, self.interpolator_pos_ion)
    
    
    def __call__(self, d_m, pos_mobility, neg_mobility):#, charges=None):
        ''' Evaluate the charging probabilities for the given input parameters.
        The input d_m can be a vector.
        '''
        
        # scale
        log_diam = np.log10(d_m)
        scaled_pos_mob = pos_mobility * 1e4
        scaled_neg_mob = neg_mobility * 1e4
        
        average_flux_coefficients = np.zeros((2, self.n_charges, d_m.shape[0]))
        for idx in range(self.n_charges):
            average_flux_coefficients[0, idx] = self.interpolator[0][idx](
                (log_diam, scaled_pos_mob, scaled_neg_mob), method='linear'
                )
            average_flux_coefficients[1, idx] = self.interpolator[1][idx](
                (log_diam, scaled_pos_mob, scaled_neg_mob), method='linear'
                )
        
        return 10**average_flux_coefficients  # Scale back
        # return average_flux_coefficients  # Scale back
    
    
    def save_interpolators(self, fname):
        with open(fname, 'wb') as outp:
            pickle.dump(self.interpolators, outp, pickle.HIGHEST_PROTOCOL)
    
    
    def load_interpolators(self, fname):
        with open(fname, 'rb') as inp:
            self.interpolator = pickle.load(inp)



class WiedensohlerChargingModel():
    
    
    def __init__(self,
                 particle_radius,
                 particle_charges_output,
                 temperature=None,
                 ):
        
        self.particle_radius = particle_radius
        
        # The charges whose probabilities will be returned
        self.particle_charges_out = particle_charges_output
        
        if temperature:
            assert temperature > 270 and temperature < 370
            self.temperature = temperature
        else:
            self.temperature = 298.15  # Kelvins
        
    
    def charging_probability(self):
        
        # Ratio of positive to negative ion mobilities
        Z_ratio = 0.875
        
        # Approximation coefficients
        a = np.array([[-26.3328, -2.3197, -0.0003, -2.3484, -44.4756],
                      [ 35.9044,  0.6175, -0.1014,  0.6044,  79.3772],
                      [-21.4608,  0.6201,  0.3073,  0.4800, -62.8900],
                      [  7.0867, -0.1105, -0.3372,  0.0013,  26.4492],
                      [ -1.3088, -0.1260,  0.1023, -0.1553, -5.7480],
                      [  0.1051,  0.0297, -0.0105,  0.0320,  0.5049]])
        
        # Preallocate
        cp = np.zeros((self.particle_charges_out.shape[0], self.particle_radius.shape[0]))
        
        log_dp = np.log10(self.particle_radius * 2 / 1e-9)[:, np.newaxis]
        for N_idx, N in enumerate(self.particle_charges_out):
            # N = amount of charge (can be negative)
            # N_idx = place of that N in the matrix
            
            assert np.issubdtype(N, np.integer)
            
            if np.abs(N) <= 2:
                cp[N_idx] = 10**(np.sum(a[:, [N + 2]].T * log_dp**np.arange(6), axis=1))
            
            else:
                common_term = (2 * np.pi * VACUUM_PERMITTIVITY * 2 * self.particle_radius
                               * BOLTZMANN_CONSTANT * self.temperature
                               / ELECTRON_CHARGE**2)
                cp[N_idx] = (ELECTRON_CHARGE
                     / np.sqrt(4 * np.pi**2 * VACUUM_PERMITTIVITY * self.temperature
                               * 2 * self.particle_radius * BOLTZMANN_CONSTANT)
                     * np.exp(-(N - common_term * np.log(Z_ratio))**2
                              / (2 * common_term))
                     )
        
        return cp
