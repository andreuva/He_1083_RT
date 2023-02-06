# He 1083 forward modelling with radiative transfer
This repository contains the code needed to do the modelling of the He 1083 with relaxed flat spectrum aproximation in a slab with radiative transfer.

We include different quadratures to do the integration of the radiative transfer equation. We suggest to use the 64x16 or pl13n100 for best performance and 16x4 for speed.

To run a simple test, change the `parameters.py` file to your liking and run the `main.py` file. The code will run the model and save the results in the `output_*/` folder.

