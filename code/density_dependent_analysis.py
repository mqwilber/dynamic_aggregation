"""
Description
-----------
Script calculates the boundary at which parasites fail regulate host populations
under the feasible set model for parasite aggregation. Results are saved to
pickles files and used in summary.ipynb to make plots.

Author: Mark Wilber

"""

from __future__ import division
from PyDSTool import *
import numpy as np
import matplotlib.pyplot as plt
import dynamic_top_down_functions as df
import pandas as pd

def build_density_dependent_ode(params, 
                                k2_moment="(P / H) + ((P/ H)**2 * (k + 1)/k)",
                                death_rate="d",
                                birth_rate="b",
                                time=(0, 20), 
                                init_cond=(10, 10)):
    """
    Build a density dependent ode given some parameters and return the ODE

    Parameters
    ----------
    params : dict
        Dictionary of parameters

    k2_moment : str
        The 2nd moment of interest.
        - "(P / H) + ((P/ H)**2 * (k + 1)/k)": Negative Binomial (Default)
        - "(P / H) + (P / H)**2": Poisson
        - "(((1 - (1 / H)) * (P / H) * (k + (P / H))) / (k + (1 / H))) + (P / H)**2": Finite Negative Binomial
        - "mom2" : Arbitrary 2nd moment
    death_rate: str
        Either
        - Density-independent death: "d"
        - Density-dependent death: "d + (b-d)*(H/Hk)"
    birth : str
        Either
        - No effect of parasite on birth: "b"
        - Multiplicative effect of parasite on host fecundity assuming NBD: "b*((k*H) / (xi*P + k*H))**k"
    init_cond : tuple
        (H, P) initial conditions

    """

    state_variables = ('H', 'P')

    phi = "(lam*P)/(H0+H)" # Free-living stage approximation

    Hrhs = "{0}*H - ({1})*H - alpha*P".format(birth_rate, death_rate)
    Prhs = "H*({0}) - ({1} + mu)*P - alpha*H*({2})".format(phi, death_rate, k2_moment)
    rhs_eqns = (Hrhs, Prhs)
    init_cond = init_cond
    time = time

    dd_ode = df.ODE_model(params, state_variables, rhs_eqns, init_cond, time)

    return dd_ode

def sim_feas_regulation_threshold(dd_ode, alpha_vals, lambda_vals,
                            update_param=('k', 'feasible_k')):
    """
    Simulate the feasible model regulation threshold given the ODE model, some
    alpha (pathogenicity) values and some lambda
    (instantaneous birth rate of parasite infective states) values.


    """

    params = dd_ode.params
    reg_vals = []

    for lambda_val in lambda_vals:

        params['lam'] = lambda_val

        host_equils = np.empty(len(alpha_vals))

        for i, alpha_val in enumerate(alpha_vals):

            params['alpha'] = alpha_val
            dd_ode.set_params(params)

            # Set initial conditions close to the H and P equilibrium
            # if possible

            P_init, H_init = df.equilibrium_feasible(params)

            if np.any(np.array([P_init, H_init]) <= 0):

                if df.can_regulate(params):
                    P_init, H_init = df.equilibrium_nbd(params)
                else:
                    P_init, H_init = (10, 10)

            dd_ode.set_initial_conditions({'H' : H_init, 'P' : P_init})

            # Simulate to equilibrium
            dd_res_feas = dd_ode.get_discrete_trajectories(delta_t=0.005,
                                update_param=update_param)

            # Extract host equil. Last value should be at equilibrium
            host_equils[i] = dd_res_feas['H'][-1]

        # Approximate threshold for no regulation
        unregulated = np.where(host_equils > 1000)[0]

        if len(unregulated) >= 1:
            reg_vals.append(alpha_vals[unregulated[0]])
        else:
            reg_vals.append(np.nan)


    return np.array(reg_vals)


if __name__ == '__main__':

    """ Main function compares the regulatory boundary for the feasible set
    macroparasite model and the classic NBD model """

    # Step 1: Define the parameters and the ODE

    mus = [0.1, 2, 3]

    for mu in mus:
        
        params = {'b' : 3,
                  'd' : 1.,
                  'Hk' : 500.,
                  'alpha' : 0.5,
                  'lam' : 20.,
                  'H0': 10,
                  'k' : 1.,
                  'mu' : mu, 
                  'xi' : 0}

        # Build model with NBD second moment
        dd_ode_nbd = build_density_dependent_ode(params)

        # Build model with FNBD second moment
        dd_ode_fnbd = build_density_dependent_ode(params, 
                    k2_moment="(((1 - (1 / H)) * (P / H) * (k + (P / H))) / (k + (1 / H))) + (P / H)**2")

        # Step 2: Calculate the value of alpha at which the feasible system can no
        # longer regulate hosts for a variety of alphas and lambdas.

        lambda_vals = np.linspace(5, 30, num=15)
        alpha_vals = np.linspace(0.01, 12, num=54)

        # Use feasible_k or feasible_k_mle gives the same answer. Reduced region
        # in which parasites can regulate host populations.  There is a slight diffe

        print("Working on NBD...")
        reg_vals_nbd = sim_feas_regulation_threshold(dd_ode_nbd, alpha_vals, 
                                    lambda_vals, update_param=('k', 'feasible_k'))

        print("Working on FNBD...")
        reg_vals_fnbd = sim_feas_regulation_threshold(dd_ode_fnbd, alpha_vals, 
                                    lambda_vals, update_param=('k', 'feasible_k'))

        # Step 3: Compute this threshold for the NBD model as well
        nbd_reg_vals = np.empty(len(lambda_vals))

        for i, lambda_val in enumerate(lambda_vals):

            params['lam'] = lambda_val
            nbd_reg_vals[i] = df.nbd_boundary_alpha(params)

        # Step 4: Save results
        sim_results = {'parameters': params, 
                       'lambda_vals': lambda_vals,
                       'NBD_k_1_alphas': nbd_reg_vals,
                       'FEAS_w_nbd_alphas': reg_vals_nbd,
                       'FEAS_w_fnbd_alphas': reg_vals_fnbd} 
        pd.to_pickle(sim_results, "../results/regulation_boundaries_mu={0}_feas_equil.pkl".format(mu))







