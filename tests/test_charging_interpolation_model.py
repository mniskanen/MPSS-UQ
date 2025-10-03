# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as resources

from time import perf_counter
from tqdm import tqdm

from MPSS_UQ.chargingmodels import LYFChargingModel, LYFInterpolator, LYFFluxInterpolator


def test_interpolation_accuracy():
    
    d_m = np.geomspace(1e-9, 1e-6, 32)
    charges_output = np.arange(-8, 8 + 1)
    # charges_output = np.arange(-25, 25 + 1)
    
    charger_direct = LYFChargingModel(d_m / 2,
                                      charges_output,
                                      max_considered_charge=25
                                      )
    
    # Compute charging probability using the flux interpolator
    fname = resources.files('MPSS_UQ.data') / 'ion_flux_interpolator_data.npz'
    flux_interpolator = LYFFluxInterpolator(fname)
    charger_flux_interpolator = LYFChargingModel(d_m / 2,
                                                 charges_output,
                                                 flux_interpolator=flux_interpolator
                                                 )
    
    # Compute charging probability with 'direct' interpolation
    fname = resources.files('MPSS_UQ.data') / 'charging_prob_interpolator_data.npz'
    charger_interpolator = LYFInterpolator(fname)
    
    n_tests = 1  # 10
    time_direct = 0
    time_interp_1 = 0
    time_interp_2 = 0
    for _ in tqdm(range(n_tests)):
        
        # Choose test parameters
        while True:
            pos_ion_mobility = np.random.uniform(low=1.05e-4, high=1.70e-4)  # 1.20e-4
            neg_ion_mobility = np.random.uniform(low=1.05e-4, high=2.10e-4)  # 1.35e-4
            ion_ratio = 1#np.random.normal(loc=1.0, scale=0.2/2)
            
            if pos_ion_mobility < neg_ion_mobility:
                break
        
        # Run the direct model
        t1 = perf_counter()
        cp_direct = charger_direct.charging_probability(pos_ion_mobility,
                                                        neg_ion_mobility,
                                                        ion_ratio,
                                                        )
        t2 = perf_counter()
        
        # Run the flux interpolation model
        cp_interp_1 = charger_flux_interpolator.charging_probability(pos_ion_mobility,
                                                                     neg_ion_mobility,
                                                                     ion_ratio,
                                                                     )
        t3 = perf_counter()
        
        # Run the 'direct' interpolator
        cp_interp_2 = charger_interpolator(d_m,
                                           pos_ion_mobility,
                                           neg_ion_mobility,
                                           charges_output
                                           )
        t4 = perf_counter()
        
        time_direct += t2 - t1
        time_interp_1 += t3 - t2
        time_interp_2 += t4 - t3
        
        # Calculate the errors
        rel_error_1 = 0
        rel_error_2 = 0
        mean_abs_error_1 = 0
        mean_abs_error_2 = 0
        for i in range(charges_output.shape[0]):
            rel_error_1 += \
                np.linalg.norm(cp_direct[i] - cp_interp_1[i]) / np.linalg.norm(cp_direct[i])
            rel_error_2 += \
                np.linalg.norm(cp_direct[i] - cp_interp_2[i]) / np.linalg.norm(cp_direct[i])
            mean_abs_error_1 += np.mean(np.abs(cp_direct[i] - cp_interp_1[i]))
            mean_abs_error_2 += np.mean(np.abs(cp_direct[i] - cp_interp_2[i]))
        avg_rel_error_1 = rel_error_1 / charges_output.shape[0]
        avg_rel_error_2 = rel_error_2 / charges_output.shape[0]
        avg_mean_abs_error_1 = mean_abs_error_1 / charges_output.shape[0]
        avg_mean_abs_error_2 = mean_abs_error_2 / charges_output.shape[0]
        print(f'\nAverage relative error (flux interp): {avg_rel_error_1 * 100 : .2g} %')
        print(f'\nAverage relative error (cp interp): {avg_rel_error_2 * 100 : .2g} %')
        print(f'Average mean absolute error (flux interp): {avg_mean_abs_error_1 * 100 : .2g} %\n')
        print(f'Average mean absolute error (cp interp): {avg_mean_abs_error_2 * 100 : .2g} %\n')
        
        # plot to compare charged fractions
        plt.figure(num=3), plt.clf()
        plt.title('Steady-state charge distribution, flux interpolator')
        plt.xlabel('Particle diameter (m)')
        for idx, k in enumerate(charges_output):
            if k < 0:
                label = 'neg. charges, direct' if k == -1 else None
                plt.loglog(d_m, cp_direct[idx], 'r-', label=label)
                label = 'neg. charges, interp' if k == -1 else None
                plt.loglog(d_m, cp_interp_1[idx], 'k--', label=label)
            elif k == 0:
                plt.loglog(d_m, cp_direct[idx], 'm-', label='0, direct')
                plt.loglog(d_m, cp_interp_1[idx], 'k--', label='0, interp')
            elif k > 0:
                label = 'pos. charges, direct' if k == 1 else None
                plt.loglog(d_m, cp_direct[idx], 'b-', label=label)
                label = 'pos. charges, interp' if k == 1 else None
                plt.loglog(d_m, cp_interp_1[idx], 'k--', label=label)

        plt.legend()
        plt.pause(0.1)
        
        plt.figure(num=4), plt.clf()
        plt.title('Steady-state charge distribution, charge prob. interpolator')
        plt.xlabel('Particle diameter (m)')
        for idx, k in enumerate(charges_output):
            if k < 0:
                label = 'neg. charges, direct' if k == -1 else None
                plt.loglog(d_m, cp_direct[idx], 'r-', label=label)
                label = 'neg. charges, interp' if k == -1 else None
                plt.loglog(d_m, cp_interp_2[idx], 'k--', label=label)
            elif k == 0:
                plt.loglog(d_m, cp_direct[idx], 'm-', label='0, direct')
                plt.loglog(d_m, cp_interp_2[idx], 'k--', label='0, interp')
            elif k > 0:
                label = 'pos. charges, direct' if k == 1 else None
                plt.loglog(d_m, cp_direct[idx], 'b-', label=label)
                label = 'pos. charges, interp' if k == 1 else None
                plt.loglog(d_m, cp_interp_2[idx], 'k--', label=label)

        plt.legend()
        plt.pause(0.1)
        
        assert avg_mean_abs_error_1 < 1e-3, \
            'Average mean absolute error between direct and flux-interpolated solutions too large'
        assert avg_mean_abs_error_2 < 1e-3, \
            'Average mean absolute error between direct and cp-interpolated solutions too large'
    
    print(
        f'\nTimings for direct model: {time_direct : .2e} s, ' + 
        f'and flux interp model: {time_interp_1 : .2e} s, i.e., ' +
        f'and charge prob. interp model: {time_interp_2 : .2e} s, i.e., ' +
        f'the charge prob. interp model was {time_direct / time_interp_2 : .0f} times faster ' +
        'than the direct one.'
        )
    

if __name__ == '__main__':
    test_interpolation_accuracy()
