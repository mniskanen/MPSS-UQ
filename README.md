
# MPSS-UQ 1.0

**Bayesian inversion of mobility particle size spectrometer data with uncertainty quantification**

MPSS-UQ is a Python package for estimating aerosol particle size distributions (PSDs) from mobility particle size spectrometer (MPSS) measurements, such as DMPS or SMPS data, in a Bayesian framework. It provides posterior estimates of the PSD together with credible intervals that quantify the uncertainty in the result.

A key feature of MPSS-UQ is the ability to **propagate the uncertainty in the bipolar charging probability** into the PSD estimates. This is done by treating the charger ion mobilities as nuisance parameters and marginalizing them over their physically plausible range, using a modular Bayesian approach. The result is a set of posterior samples from which credible intervals and derived quantities (with uncertainties) can be computed.

MPSS-UQ is described in detail in the accompanying paper:

> Niskanen, M., Ursin, A., Ylisirniö, A., and Lehtinen, K. E. J.: Quantifying charger-related uncertainty in electrical mobility analysis of aerosols with MPSS-UQ 1.0, submitted for review.

## Features

- Bayesian inversion of MPSS (DMPS/SMPS) data with posterior credible intervals (highest density intervals).
- Marginalization of charging probability over uncertain ion mobilities using the López-Yglesias–Flagan (LYF) charging model, with a fast interpolation-based approximation.
- Uncertainty propagation to derived quantities (e.g., total number concentration, mode-specific concentrations, condensation sink) via posterior samples. Custom derived quantities can be added by the user.
- Parallel processing of multi-scan datasets with automatic cache optimization for varying temperature and pressure.
- Automatic handling of multiply charged particles (configurable maximum charge state).
- DMA transfer function (Stolzenburg diffusive model), sampling line losses (Gormley–Kennedy),
  and CPC counting efficiency parametrized by user-specified instrument properties.
- Built-in plotting functions for PSD estimates, credible intervals, time series, and uncertainty maps.

## Installation

### Quick install (if you already have conda and Python ≥ 3.9)

```bash
conda create -n mpss-uq python=3.11
conda activate mpss-uq
git clone https://github.com/mniskanen/MPSS-UQ.git
cd MPSS-UQ
pip install .
```

You are ready to go — see [Quick start](#quick-start) or the scripts in `examples/`.

### Step-by-step install

<details>
<summary><b>Click to expand detailed instructions</b></summary>

These instructions walk through the full setup starting from a fresh computer. If any of the steps below are already done on your system, skip them.

#### 1. Install Miniconda

Miniconda is a lightweight Python distribution that includes `conda`, a tool for managing Python environments and packages.

1. Go to [https://docs.anaconda.com/miniconda/](https://docs.anaconda.com/miniconda/)
2. Download the installer for your operating system (Windows, macOS, or Linux).
3. Run the installer and follow the prompts. The default settings are fine.

After installation, open a terminal:
- **Windows:** Open "Anaconda Prompt" (search for it in the Start menu). Do **not** use the regular Command Prompt unless you have added conda to your PATH.
- **macOS / Linux:** Open a regular terminal. If `conda` is not recognized, you may need to close and reopen the terminal, or run `source ~/miniconda3/bin/activate`.

Verify that conda is available:

```bash
conda --version
```

This should print something like `conda 25.x.x`. If you get an error, revisit the Miniconda installation instructions.

#### 2. Download MPSS-UQ

If you have `git` installed:

```bash
git clone https://github.com/mniskanen/MPSS-UQ.git
```

If you do not have `git`, you can download the code as a ZIP file:

1. Go to [https://github.com/mniskanen/MPSS-UQ](https://github.com/mniskanen/MPSS-UQ)
2. Click the green **"Code"** button, then **"Download ZIP"**.
3. Extract the ZIP file to a location of your choice.

#### 3. Create a Python environment and install MPSS-UQ

In the terminal, navigate to the folder where you downloaded MPSS-UQ:

```bash
cd MPSS-UQ
```

(Replace `MPSS-UQ` with the actual path if you extracted the ZIP to a different location, e.g., `cd C:\Users\YourName\Downloads\MPSS-UQ-main`.)

Create a new conda environment with Python 3.11:

```bash
conda create -n mpss-uq python=3.11
```

When prompted, type `y` and press Enter. Then activate the environment:

```bash
conda activate mpss-uq
```

Your terminal prompt should now show `(mpss-uq)` at the beginning. Finally, install MPSS-UQ and its dependencies:

```bash
pip install .
```

Note the dot (`.`) at the end which tells pip to install from the current folder.

#### 4. Verify the installation

Start Python and try importing the package:

```bash
python -c "from MPSS_UQ.inversion import invert_psd; print('MPSS-UQ installed successfully')"
```

If you see `MPSS-UQ installed successfully`, the installation is complete.

#### 5. Run an example

To run the simulated data example:

```bash
cd examples
python invert_simulated_data.py
```

This will generate synthetic DMPS data, invert it, and display plots of the estimated size distribution with credible intervals. To run the real data example:

```bash
cd examples/realdata_UEF_test
python invert_real_data.py
```

</details>


## Quick start

The minimal workflow for inverting a single simulated MPSS measurement.
This assumes you are running from the `examples/` folder, where `DMPS_properties.yaml` is located (see [Instrument configuration](#instrument-configuration) for details on this file).

```python
import yaml
import numpy as np
from MPSS_UQ.particlesizers import MobilityParticleSizeSpectrometer
from MPSS_UQ.inversion import invert_psd
from MPSS_UQ.measurement_data import generate_DMPS_measurement

# 1. Load instrument configuration
with open("DMPS_properties.yaml", "r") as f:
    DMPS_prop = yaml.safe_load(f)["UEF-A20"]

# Set the nominal diameters (measurement channels) of the DMPS
DMPS_prop["d_m_data"] = np.geomspace(10e-9, 800e-9, num=30)

# Choose the charging model (see "Charging models" section below)
DMPS_prop["charging_model"] = "LYF-interp"

# Maximum number of multiple charges to model
DMPS_prop["max_charge"] = 10

# Generate synthetic DMPS measurement
# 'scenario' selects a predefined aerosol size distribution. Options include:
# Urban, Marine, Rural, Remote continental (and others; see invert_simulated_data.py for the full list)
measurement = generate_DMPS_measurement(
    DMPS_prop, scenario="Urban",
    pos_ion_mobility=1.35e-4, neg_ion_mobility=1.60e-4,
)

# Set up the inversion model
DMPS = MobilityParticleSizeSpectrometer(DMPS_prop, n_bins=70)
DMPS.set_operating_conditions(measurement.temperature, measurement.pressure)
DMPS.set_charger_properties(1.35e-4, 1.60e-4)

# Invert
psd_posterior = invert_psd(DMPS, measurement)                                      # fixed charging
psd_posterior_marg = invert_psd(DMPS, measurement, marginalize_ion_mobility=True)  # marginalized

# Get the posterior median and 95% credible intervals
median, CI_lower, CI_upper = psd_posterior_marg.summary(coverage=95)
```

## Instrument configuration

Instrument properties are specified in a YAML file. An example configuration (`DMPS_properties.yaml`) is provided in the `examples/` folder. When loading the file, the path must be correct relative to your working directory. For instance, if you run a script from the `examples/` folder, `open("DMPS_properties.yaml", "r")` will work directly. If you run from elsewhere, adjust the path accordingly. The file defines:

| Parameter | Description |
|---|---|
| `Qsh`, `Qe`, `Qa`, `Qc` | DMA flow rates (L/min) |
| `R1`, `R2`, `L` | DMA geometry: inner radius, outer radius, length (m) |
| `L_eff` | Effective DMA diffusion length (m) |
| `center_voltage_sign` | Polarity of the DMA centre electrode |
| `CPC_Qsample` | CPC sample flow rate (L/min) |
| `CPC_measuring_time` | Measurement time per channel (s) |
| `CPC_output_type` | Output type of the CPC (`counts` or `concentration`) |
| `CPC_a`, `CPC_b`, `CPC_dp50` | CPC counting efficiency parameters |
| `inlets` | Sampling line sections as `[length (m), flow rate (L/min)]` pairs |

A custom CPC counting efficiency function can also be provided programmatically; see the example scripts for details.

## Charging models

MPSS-UQ supports the following charging models, selected via the `charging_model` key in the configuration:

| Model name | Description |
|---|---|
| `Wiedensohler` | Wiedensohler (1988) approximation. Fast, but does not support varying ion properties. |
| `LYF-interp` | Interpolated LYF model. Fast and recommended for general use and marginalization over ion mobilities. Assumes ion concentration ratio = 1. |
| `LYF-interp-flux` | Interpolated LYF model at the flux coefficient level. Supports varying both ion mobilities and the ion concentration ratio. |
| `LYF-direct` | Direct evaluation of the LYF model. Slow, mainly useful for verification. |

## Example scripts

### `examples/invert_simulated_data.py`

Example with synthetic data. Demonstrates:

- Loading an instrument configuration from YAML
- Generating a simulated DMPS measurement for a chosen aerosol scenario
- Setting up the prior (Gaussian smoothness prior)
- Inverting with a fixed charging probability (Laplace approximation)
- Inverting with marginalization over ion mobilities
- Plotting PSD estimates with credible intervals
- Computing and plotting total particle number histograms
- Plotting data fit diagnostics

### `examples/realdata_UEF_test/invert_real_data.py`

Example using real DMPS field data (included as a compressed file). Demonstrates:

- Loading raw measurement data from a tab-separated file
- Converting concentrations to counts
- Creating a `MeasurementDataset` with timestamps, temperatures, and pressures
- Batch inversion of a multi-day dataset with `invert_dataset()`
- Selecting time subsets with `between_times()`
- Plotting time series of the PSD, relative uncertainty, and total particle number
- Plotting individual scans with credible intervals
- Computing derived quantities (mode concentrations, condensation sink, geometric mean diameter, etc.)

## API reference

### `MobilityParticleSizeSpectrometer(properties, n_bins=None, inversion_grid=None)`

Creates the MPSS forward model from a dictionary of instrument properties.

- `set_operating_conditions(temperature, pressure)` — updates temperature- and pressure-dependent quantities (transfer function, diffusion losses)
- `set_charger_properties(pos_mobility, neg_mobility, ion_ratio=1.0)` — sets the charger ion properties

### `invert_psd(sizer, measurement, prior=None, marginalize_ion_mobility=False, ...)`

Inverts a single MPSS measurement. Returns a `PSDPosterior` object.

Key arguments:
- `marginalize_ion_mobility=True` — marginalizes over ion mobilities using the cut posterior formulation
- `marginalization_grid` — `'standard'` (default, ~400 grid points) or `'fine'` (denser grid)
- `num_samples` — number of posterior samples (default: 5000 for marginalization)
- `use_mcmc=True` — uses MCMC sampling instead of the Laplace approximation (non-marginalized only)

### `invert_dataset(sizer, dataset, parallel=True, ...)`

Inverts a batch of measurements. Returns a `PSDPosteriorSeries` object. Accepts the same keyword arguments as `invert_psd`, plus:

- `parallel=True` — distributes inversions across CPU cores
- `sort_for_cache=True` — groups measurements by temperature/pressure for better cache performance

### `PSDPosterior`

Stores the posterior for a single measurement.

- `.summary(coverage=95)` — returns `(median, CI_lower, CI_upper)`
- `.propagate_to(func)` — propagates posterior samples through any derived quantity
- `.get_samples(num=None)` — returns posterior samples in linear (N) space
- `.set_reporting_range('measured')` or `.set_reporting_range('full')` — controls which size range is reported

### `PSDPosteriorSeries`

Stores posteriors for a time series of measurements.

- `.summary(coverage=95)` — returns arrays of `(medians, CI_lower, CI_upper)` for all scans
- `.propagate_to(func)` — propagates uncertainty for all scans
- `.between_times(start, end)` — returns a subset by time range
- Supports indexing: `series[i]` returns a `PSDPosterior`, `series[start:stop]` returns a new `PSDPosteriorSeries`

### `MeasurementDataset(datetimes, d_m_data, MPSS_outputs, output_type, temperatures, pressures)`

Container for a time series of MPSS measurements. Required input to `invert_dataset()`.

- `datetimes` — array of measurement timestamps
- `d_m_data` — array of nominal mobility diameters (m)
- `MPSS_outputs` — 2D array of CPC outputs, shape `(n_scans, n_channels)`
- `output_type` — `'counts'` or `'concentration'`
- `temperatures` — array of temperatures (K) per scan
- `pressures` — array of pressures (Pa) per scan
- `.between_times(start, end)` — returns a subset by time range
- `.at_time(datetime)` — returns the single `Measurement` closest to the given time

### `generate_DMPS_measurement(DMPS_properties, scenario, pos_ion_mobility, neg_ion_mobility, rng_seed=None)`

Generates a synthetic MPSS measurement for testing. Returns a `Measurement` object
that can be passed directly to `invert_psd()`.

### `smoothness_prior(d_m, mean=0.0, standard_deviation=1.5, correlation_length=0.5)`

Constructs a Gaussian squared-exponential smoothness prior. Parameters are defined in log₁₀ space:

- `mean` — prior mean of log₁₀(dN/dlog d_p)
- `standard_deviation` — controls the allowed range of concentrations
- `correlation_length` — in decades of diameter; controls smoothness

### Derived quantities (in `MPSS_UQ.analysis`)

All functions take posterior samples as their first argument and can be passed to `PSDPosterior.propagate_to()`:

- `total_concentration` — total number concentration
- `concentration_in_range` — number concentration in a specified diameter range
- `geometric_mean_diameter`, `mode_diameter`, `median_diameter`
- `surface_area_concentration`, `volume_concentration`
- `condensation_sink` — sulfuric acid vapour condensation sink
- `effective_diameter`, `geometric_std`
- `relative_hdi_width` — relative width of the highest density credible interval

## Reproducing the paper results

The script `scripts/create_paper_figures.py` reproduces all figures and numerical results presented in the paper. It requires the real measurement dataset included in `examples/realdata_UEF_test/`. To run:

```bash
cd scripts
python create_paper_figures.py
```

The `scripts/` folder also contains `create_LYFInterpolator.py`, which regenerates
the precomputed LYF interpolation data stored in `src/MPSS_UQ/data/`.

## Hardware and software requirements

MPSS-UQ runs on any platform that supports Python ≥ 3.9 (tested on Windows and Linux). All results in the accompanying paper were computed on a standard laptop (Intel Core i7-12700H, 32 GB RAM) in approximately 10 minutes for the full month-long dataset with marginalization.

## License

Copyright 2024–2026 Matti Niskanen.

MPSS-UQ is made available under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for details.

## Citation

If you use MPSS-UQ in your work, please cite (to be updated):

```bibtex
@article{niskanen2026mpss-uq,
  title   = {Quantifying charger-related uncertainty in electrical mobility
             analysis of aerosols with {MPSS-UQ} 1.0},
  author  = {Niskanen, Matti and Ursin, Aku and Ylisirni{\"o}, Arttu
             and Lehtinen, Kari E. J.},
  journal = {},
  year    = {2026},
  note    = {Submitted}
}
```
