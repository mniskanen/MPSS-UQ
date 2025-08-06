# -*- coding: utf-8 -*-

from MPSS_UQ.chargingmodels import LYFInterpolator, LYFFluxInterpolator


''' Just a helper script to create the LYF model interpolators. '''

# charging_model = LYFInterpolator()
charging_model = LYFFluxInterpolator()
charging_model.construct_interpolators()
charging_model.save_interpolators('interpolator_flux')
