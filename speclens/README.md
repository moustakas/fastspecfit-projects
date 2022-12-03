# Spectroscopic Lens Project using fastspecfit

## Environment Setup
To setup the environment on nersc to run this simply run the command `./fastspecfit-env-nersc`. This will create a conda environment for running other scripts.

## Analysis
For this project, we focused on one target that we identified by crossmatching known lenses from the masterlens database to all DESI targets in the Fuji and Guadalupe productions using `crossmatch_known_lenses.py`. We then visually inspected the spectroscopic lenses for promising spectroscopic lenses to perform our analysis. We identified one and worked to separate it. To run the separation pipeline, simply use the command `./speclens`. Before doing this, you either need to download the masterlens database to this directory or reach out to the authors for the file they used in their own analysis.