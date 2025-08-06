# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter
from tqdm import tqdm

from MPSS_UQ.chargingmodels import LYFChargingModel, LYFInterpolator, LYFFluxInterpolator
from MPSS_UQ.definitions import INTERPOLATOR_DIR


def test_interpolation_accuracy():
    
    d_m = np.geomspace(1e-9, 1e-6, 32)
    charges_output = np.arange(-8, 8 + 1)
    # charges_output = np.arange(-25, 25 + 1)
    
    charger_direct = LYFChargingModel(d_m / 2,
                                      charges_output,
                                      max_considered_charge=25
                                      )
    
    flux_interpolator = LYFFluxInterpolator(f'{INTERPOLATOR_DIR}/interpolator_flux_60dm_307')
    charger_interpolator = LYFChargingModel(d_m / 2,
                                            charges_output,
                                            flux_interpolator=flux_interpolator
                                            )
    
    n_tests = 3  # 10
    time_direct = 0
    time_interp = 0
    for _ in tqdm(range(n_tests)):
        
        # Choose test parameters
        while True:
            pos_ion_mobility = np.random.uniform(low=1.05e-4, high=1.70e-4)  # 1.20e-4
            neg_ion_mobility = np.random.uniform(low=1.05e-4, high=2.10e-4)  # 1.35e-4
            ion_ratio = np.random.normal(loc=1.0, scale=0.2/2)
            
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
        cp_interp = charger_interpolator.charging_probability(pos_ion_mobility,
                                                              neg_ion_mobility,
                                                              ion_ratio,
                                                              )
        
        t3 = perf_counter()
        
        time_direct += t2 - t1
        time_interp += t3 - t2
        
        # Calculate the errors
        # TODO: Actually, I think relative errors don't make sense here because
        # the scale can go down to 1e-16 where the interpolator probably works (in absolute
        # terms) worse than near 1e0, so the relative error would be comparatively huge although
        # the absolute error wouldn't matter any more than the same absolute error with values
        # closer to 1e0.
        rel_error = 0
        mean_abs_error = 0
        for i in range(charges_output.shape[0]):
            rel_error += np.linalg.norm(cp_direct[i] - cp_interp[i]) / np.linalg.norm(
                cp_direct[i]
                )
            # abs_error += np.linalg.norm(cp_direct[i] - cp_interp[i], ord=2)
            mean_abs_error += np.mean(np.abs(cp_direct[i] - cp_interp[i]))
        avg_rel_error = rel_error / charges_output.shape[0]
        avg_mean_abs_error = mean_abs_error / charges_output.shape[0]
        print(f'\nAverage relative error: {avg_rel_error * 100 : .2g} %')
        print(f'Average mean absolute error: {avg_mean_abs_error * 100 : .2g} %\n')
        
        # plot to compare charged fractions
        plt.figure(num=3), plt.clf()
        plt.title('Steady-state charge distribution')
        plt.xlabel('Particle diameter (m)')
        for idx, k in enumerate(charges_output):
            if k < 0:
                label = 'neg. charges, direct' if k == -1 else None
                plt.loglog(d_m, cp_direct[idx], 'r-', label=label)
                label = 'neg. charges, interp' if k == -1 else None
                plt.loglog(d_m, cp_interp[idx], 'k--', label=label)
            elif k == 0:
                plt.loglog(d_m, cp_direct[idx], 'm-', label='0, direct')
                plt.loglog(d_m, cp_interp[idx], 'k--', label='0, interp')
            elif k > 0:
                label = 'pos. charges, direct' if k == 1 else None
                plt.loglog(d_m, cp_direct[idx], 'b-', label=label)
                label = 'pos. charges, interp' if k == 1 else None
                plt.loglog(d_m, cp_interp[idx], 'k--', label=label)

        # plt.axis([2e-10, 1e-5, 1e-4, 1.15])
        plt.legend()
        # plt.draw()
        plt.pause(0.1)
        
        assert avg_mean_abs_error < 1e-3, \
            'Average mean absolute error between direct and interpolated solutions too large'
    
    print(
        f'\nTimings for direct model: {time_direct : .2e} s, ' + 
        f'and interp model: {time_interp : .2e} s, i.e., ' +
        f'the interp model was {time_direct / time_interp : .0f} times faster.'
        )
    

if __name__ == '__main__':
    test_interpolation_accuracy()
