# He 1083 forward modelling with radiative transfer

This repository contains the code needed to do the modelling of the He 1083 with relaxed flat spectrum aproximation in a slab with radiative transfer.
The code is described in the paper "" (include link). And was mainly developed by Andres Vicente and Tanausu del Pino Aleman.

## Requirements

The requirements for this code are quite simple, you need to have python 3.6 or higher and the following packages:
- numpy
- scipy
- tqdm

you can install them with pip:
```
pip install numpy scipy tqdm
```

## Usage

To run a simple test, change the `parameters.py` file to your liking and run the `main.py` file. The code will run the model and save the results in the `output_*/out/` folder.
In that folder, you will find the following files:
- `MRC` : Maximum relative change of each iteration in populations and coherences
- `parameters.out` : Parameters used in the run
- `tau_*.out` : Optical depth of the output ray number *
- `stokes_*.out` : Stokes parameters of the output ray number *

We included different quadratures to do the integration of the radiative transfer equation. We suggest to use the 64x16 or pl13n100 quadratures for best performance and 16x4 for speed.

To relax the flat spectrum approximation you can put the `especial=True` flag on the `parameters.py` file. This will use the different radiation fields for each component of the He 1083 multiplet.

