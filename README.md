This repository contains the scripts necessary to replicated the analyses found in the manuscript *Dynamic parasite aggregation reduces parasite regulation of host populations and the stability of host-parasite interactions*.  The repository is organized as follows

`code/`:

-`summary.ipynb`: This notebook contains the most of the analyses described in the main text.

1. This notebook is used to calculate the interpolated feasible surface. Note that this interpolated feasible surface must be calculated before any of the dynamic models can be run.  This is done in section **Incorporating the feasible set/partition model** and the results are saved to the file `interp_grid_sigma75_flat.pkl`, which should live in the `code` directory.

2. This notebook plots the predictions regarding how Feasible k and Fixed k models differ in their predictions of parasite regulation. The script `density_dependent_analysis.py` is run first to get the Feasible k predictions.

3. The stability analysis is performed for the Fixed k and Feasible k models. 

4. Additional plots are made that show the behavior of the feasible set and the model dynamics.

-`density_dependent_analysis.py`: This script is run in `summary.ipynb` and simulates the Feasible k Model to calculate where this model predicts parasite regulation of host populations.

-`dynamic_top_down_functions.py`: A variety of different functions used in the analyses.

`docs/`: 
    
-`dynamic_top_down_manuscript.pdf`: A copy of the manuscript.

`results/`: 

-`simulated_fs_metrics.pkl`: A pickled files that contains the simulated feasible surface without any interpolation.
