This repository contains the scripts necessary to replicate the analyses found in the manuscript *Dynamic parasite aggregation reduces parasite regulation of host populations and the stability of host-parasite interactions*.  The repository is organized as follows

`code/`:

-`manuscript_analyses.ipynb`: This notebook contains the analyses described in the main text.

1. This notebook is used to calculate the partition model variance to mean surface.

2. This notebook plots the predictions regarding how the Feasible Aggregation and Fixed k Models differ in their predictions of parasite regulation.
3. The stability analysis is performed for the Fixed k and Feasible k models. 

4. Additional plots are made that show the behavior of the dynamics of host-parasite interactions under the partition model assumption.

-`macroparasites.py`: Functions for analyzing the dynamics of host-parasite interactions with Feasible Aggregation and Fixed k aggregation.
    
-`dynamic_top_down_manuscript.pdf`: A copy of the manuscript.

`results/`: 

-`simulated_fs_metrics.pkl`: A pickled file that contains the simulated feasible surface without any interpolation.
