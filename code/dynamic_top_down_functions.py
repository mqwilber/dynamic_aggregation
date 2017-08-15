from __future__ import division
import numpy as np
import pandas as pd
from PyDSTool import *
from sympy import symbols
from scipy.optimize import fsolve
from scipy.misc import derivative
from descartes import PolygonPatch
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
from shapely import geometry
import math
import scipy.stats as stats
import macroeco.models as md

# Check for pypartitions
try:
    import pypartitions as pyp # Package obtained from https://github.com/klocey/partitions
except ImportError:
    print("Feasible set package not found. Download from\n" +
                "https://github.com/klocey/partitions and set to PYTHONPATH")


# Global variables for feasible grid
INTERP_VALS = np.linspace(3, 500, num=10000) # These dimensions need to match INTERP_GRID
INTERP_GRID = pd.read_pickle("interp_grid_sigma75_flat.pkl")
#INTERP_GRID_MLE = pd.read_pickle("interp_grid_sigma20_mle.pkl")
N = len(INTERP_VALS)


class ODE_model:
    """
    Build an ODE model.  This is a wrapper for PyDSTools to make it a bit
    easier to specify the model and get the trajectories

    Parameters
    ----------
    params : dict
        Dictionary of parameters in the form key (str): value (float) where
        key is the parameter name and value is the value of the parameter
    state_variables : tuple
        Tuple of strings where each string is the name of a state variable
    rhs_eqns : tuple
        A tuple containing the RHS of the odes corresponding to the state
        variables given in state_variables.
    init_cond: tuple
        A tuple of initial conditions corresponding to the state variables
    time : list
        A list of length two with a starting time (time[0]) and an ending time
        (time [1]) over which the ODE should be integrated
    boundaries : tuple
        tuple 2 item lists giving the boundaries for the state variables

    """

    def __init__(self, params, state_variables, rhs_eqns, init_conds, time,
                    boundaries=None):

        assert len(state_variables) == len(rhs_eqns), \
            "Need the same number of state variables and ODEs equations"

        assert len(state_variables) == len(init_conds), \
            "Need the same number of state variables and initial conditions"


        self.params = params
        self.state_variables = state_variables
        self.rhs_eqns = rhs_eqns
        self.boundaries = boundaries
        self.init_conds = dict(zip(state_variables, init_conds))

        # Assign all parameters
        for key, val in params.iteritems():
            exec("{0} = Par({1}, '{0}')".format(key, val))

        # Put all parameters in a list
        par_list = []
        for key in params.iterkeys():
            par_list.append(eval(key))

        # Assign the state variables
        for variable in state_variables:
            exec("{0} = Var('{0}')".format(variable))

        # Make dictionary of ODEs
        rhs_dict = {}
        for sv, rhs in zip(state_variables, rhs_eqns):
            rhs_dict[sv] = eval(rhs) # Save equations as dict

        DSargs = args(name="model")
        DSargs.pars = par_list
        DSargs.varspecs = args(**rhs_dict)
        DSargs.ics = dict(zip(state_variables, init_conds))
        DSargs.tdata = time

        # Add state variable boundaries
        if boundaries:
            assert len(boundaries) == len(state_variables), \
            "Boundary conditions and state variables must be of same length"
            DSargs.xdomain = dict(zip(state_variables, boundaries))

        # TODO: Provide option for a different integrator
        self.ode_model = Generator.Vode_ODEsystem(DSargs)

    def get_bifurcation(self, free_params, init_conds, **kwargs):
        """
        Build a bifurcation class and compute the bifurcation curve

        Parameters
        -----------
        free_params : list
            List with strings for free parameters
        init_conds : dict
            Dict where keys are a string with state variable names and
            values are initial conditions close to the equilibrium.
        kwargs : Additional key words
            max_points, max_step, min_step, step_size, bif_points that specify
            how the bifurcation is done

        """

        # Set parameters and initial conditions
        self.set_initial_conditions(init_conds)
        self.ode_model.set(pars=self.params)

        # Make cont class and set parameters
        self.bif = ContClass(self.ode_model)
        PCargs = args(name="mod1", type="EP-C", force=True) # Equilibrium point curve
        PCargs.freepars = free_params # TODO: Multiple parameters not working
        PCargs.MaxNumPoints = kwargs.get('max_points', 100)   # The following 3 parameters are set after trial-and-error
        PCargs.MaxStepSize  = kwargs.get('max_step', 2)
        PCargs.MinStepSize  = kwargs.get('min_step', 1e-6)
        PCargs.StepSize     = kwargs.get('step_size', 0.1)
        PCargs.LocBifPoints = kwargs.get('bif_points', 'LP')
        PCargs.SaveEigen = True

        self.bif.newCurve(PCargs)
        self.bif['mod1'].backward()
        return(self.bif['mod1'])



    def get_continuous_trajectories(self):
        """ Get the continuous trajectory for the ODE """

        self.traj = self.ode_model.compute('traj')
        return(self.traj.sample())

    def set_initial_conditions(self, init_conds):
        self.init_conds = init_conds
        self.ode_model.set(ics=init_conds)

    def set_params(self, params):
        """ Set the parameters for the ODE """

        for key, value in params.iteritems():
            self.params[key] = value

        self.ode_model.set(pars=self.params)

    def set_time(self, time):
        self.ode_model.tdata = time

    def discretize_model(self):
        """
        Discretize the given model using the Euler Method
        """

        self.discretized_eqns = {}
        for sv, eqn in zip(self.state_variables, self.rhs_eqns):
            self.discretized_eqns[sv] = sv + " + delta_t * " + "(" + eqn + ")"

    def get_discrete_trajectories(self, delta_t=0.01, update_param=None):
        """
        Compute a discretized simulation using Euler's Method

        Parameters
        ----------
        delta_t : float
            Time step in Euler's Method
        update_param : tuple
            First item, str, parameter
            Second item, str, function updating that parameter

        """

        # Discretize the model
        self.discretize_model()

        # Put parameters in name space
        for key, value in self.params.iteritems():
            exec(key + "={0}".format(value))

        # Put the state variables in the name space with initial condition
        for key, value in self.init_conds.iteritems():
            exec(key + "={0}".format(value))

        # Put state variables in a tuple
        sv_tup = self.init_conds

        # Set time parameters
        time = self.ode_model.tdata[1] - self.ode_model.tdata[0]
        time_vals = np.arange(0, time + delta_t, step=delta_t)

        results = {}

        # Dictionary to model results
        for sv in self.state_variables:
            results[sv] = np.empty(len(time_vals))
            results[sv][0] = eval(sv)

        # Also save updated parameter
        if update_param:
            results[update_param[0]] = np.empty(len(time_vals))
            exec("{0} = {1}(**sv_tup)".format(update_param[0],
                                                        update_param[1]))
            results[update_param[0]][0] = eval(update_param[0])

        # Run discretized simulation
        for i in np.arange(1, len(time_vals)):

            # Perform update rule on state variables
            for sv in self.state_variables:
                results[sv][i] = eval(self.discretized_eqns[sv])

            # Reassign current state_variables
            sv_tup = {}
            for sv in self.state_variables:
                exec(sv + "={0}".format(results[sv][i]))
                sv_tup[sv] = results[sv][i]

            # Update parameter if necessary
            if update_param:
                exec("{0} = {1}(**sv_tup)".format(update_param[0],
                                                        update_param[1]))
                results[update_param[0]][i] = eval(update_param[0])

        results['t'] = time_vals

        self.results = results

        return results



def equilibrium_nbd(params):
    """
    Equilibrium for host-parasites following NBD distribution

    Parameters
    ----------
    params : same as above

    Returns
    -------
    : tuple
        (P, H) equilibrium


    Notes
    -----
    This equation doesn't match what is given in Anderson
    and May 1978.  They swapped a and b in their equation in the manuscript.


    """


    lam = params['lam']
    H0 = params['H0']
    d = params['d']
    b = params['b']
    mu = params['mu']
    alpha = params['alpha']
    k = params['k']

    holder =  alpha / (b - d)

    fact = (mu + d + alpha + ((b - d) * (k + 1) / (k)))
    num = H0 * fact
    denom = lam - fact
    H = num / denom
    P = H * (1 / holder)

    # Use non-linear solver
    if params['xi'] != 0:

        P, H = fsolve(equil_func, (P, H), args=(params, True, False))

    return((P, H))


def equilibrium_fnbd(params):
    """
    Calculate the equilibrium point for the host-parasite equation with
    a FNBD host-parasite distribution

    Parameters
    ----------
    params : dict
        Host parasite parameters as given in Andersona and May 1978
        lam = rate of production of transmission stages
        a = host birth rate
        b = host death rate
        mu = parasite death rate
        alpha = parasite pathogenecity
        k = aggregation parameter
        H0 = Determines the efficacy of infective stages infecting a host.
        If H0 is small, all infected stages infect a host.  This is similar to
        the half-saturation constant

    Returns
    -------
    : tuple
        (P, H) equilibrium


    """

    # lam = params['lam']
    # H0 = params['H0']
    # d = params['d']
    # b = params['b']
    # mu = params['mu']
    # alpha = params['alpha']
    # k = params['k']
    # xi = params['xi']

    # # Helper params
    # holder = alpha / (b - d)
    # om = d + mu + (alpha / holder)

    # # Parameters of the quadratic
    # a_coef = (k*lam*holder**2 - om*k*holder**2 - alpha*k*holder**2 - alpha*holder)
    # b_coef = (-om*H0*k*holder + lam*holder - om*holder - alpha*H0*k*holder -
    #                               alpha*H0 + alpha*k*holder + alpha)
    # c_coef = (-om*H0 + alpha*H0*k + alpha*H0*(1 / holder))

    # P1 = (-b_coef + np.sqrt(b_coef**2 - 4*a_coef*c_coef)) / (2 * a_coef)
    # P2 = (-b_coef - np.sqrt(b_coef**2 - 4*a_coef*c_coef)) / (2 * a_coef)

    # # if P1 > P2:
    # #     ret_P = P1
    # # else:
    # #     ret_P = P2

    # H1 = P1 * holder
    # P, H = (P1, H1)

    # if params['xi'] != 0:

    start = equilibrium_nbd(params)
    P, H = fsolve(equil_func, start, args=(params, True, True))


    return((P, H))


def equilibrium_feasible(params):
    """
    Given a set of params, find the equilibrium for the feasible
    host-parasite model.

    Parameters
    ----------
    params : dict
        Parameters of the host-parameter model

    Returns
    -------
    : tuple
        Equilibrium (Pstar, Hstar) from the feasible set model. 
    """

    start = equilibrium_nbd(params)
    roots = fsolve(equil_func, start, args=(params, False, False))

    return(roots)


def equil_func(x, params, fixed_k=True, finite=False):
    """
    Solves non-linear set of equilibrium equations for P and H

    Parameters
    ----------
    x : tuple
        P and H
    params : dict
        Dictionary of parameters in the model
    fixed_k : bool
        If True assumes k is fixed, otherwise it follows feasible set
    finite : If True assumes a finite NBD, otherwise an infinite NBD

    Returns
    -------
    : list
        Returns evaluated equations
    """

    P = x[0]
    H = x[1]
    mu = P / H

    if fixed_k:
        k = params['k']
    else:
        k = feasible_k(P, H)

    if finite: # Use a finite negative binomial distribution
        pgf = fnbd_pgf
        second_mom = ((1 - 1/H)*(P / H)*(k + (P / H)) / (k + 1/H)) + (P / H)**2
    else: # Use the negative binomial distribution
        pgf = lambda P, H, k, xi: ((k*H) / ((xi)*P + k*H))**k
        second_mom = (mu + mu**2*(1 + k**-1))

    eq1 = H*params['b']*pgf(P, H, k, params['xi']) - \
                H*params['d'] - params['alpha']*P
    eq2 = -(params['d'] + params['mu'])*P + \
           (params['lam']*P*H) / (params['H0'] + H) + \
           - params['alpha']*H*second_mom

    return([eq1, eq2]) 

def fnbd_pgf(P, H, k, xi):
    """ PGF for the finite negative binomials distribution """

    vals = np.arange(P + 1)
    return(np.sum(md.cnbinom.pmf(vals, mu=P / H, k_agg=k, b=P) * (1 - xi)**vals))

def can_regulate(params):
    """
    Under a (finite) negative binomial assumption, check whether the parameters
    satisfy the conditions necessary for regulation.

    Parameters
    ----------
    params : same as above

    Returns
    -------
    : bool
        Whether or not regulation can occur on the given parameters

    """

    if(params['alpha'] == 0):
        return(False)

    alpha = params['alpha']
    b = params['b']
    d = params['d']
    k = params['k']
    mu = params['mu']
    lam = params['lam']

    reg = (lam > mu + d + alpha + (b - d) + ((b - d) / k))
    return(reg)
    

def host_equil(alpha, params, equil_fxn, upper=10000):
    """
    Function to numerically calculate the threshold at which parasites can no longer regulate
    host populations

    Parameters
    -----------
    alpha : float
        Pathogenicity parameter
    params : dict
        Same dict that is passes into equil_fxn
    equil_fxn : fxn
        Either equilibrium_nbd or equilibrium_fnbd function
    upper : float
        Approximate upper bound tht dictates when host have become unregulated

    Returns
    -------
    : float
        Difference between host equilibrium and upper

    """
    params['alpha'] = alpha
    H = np.log(equil_fxn(params)[1])
    holder = H - np.log(upper)

    if np.isnan(holder):
        holder = 5 # When it explodes just set it equal to a positive number

    return holder

def nbd_boundary_lam(params):
    """
    Calculates the exact NBD boundary at which a parasite can regulate a
    host. Exact solution from Anderson and May 1978

    """
    alpha = params['alpha']
    b = params['b']
    d = params['d']
    k = params['k']
    mu = params['mu']

    lam = mu + d + alpha + (b - d) + ((b - d) / k)
    return(lam)

def nbd_boundary_alpha(params):
    """
    Calculates the exact NBD boundary at which a parasite can regulate a
    host. Exact solution from Anderson and May 1978

    """
    lam = params['lam']
    b = params['b']
    d = params['d']
    k = params['k']
    mu = params['mu']

    alpha = -1 * (mu + d + (b - d) + ((b - d) / k)) + lam
    return(alpha)

def nbd_boundary_k(params):
    """
    Calculates the exact NBD boundary at which a parasite can regulate a
    host. Exact solution from Anderson and May 1978

    """
    alpha = params['alpha']
    b = params['b']
    d = params['d']
    lam = params['lam']
    mu = params['mu']

    k = (b - d) / (lam - mu - d - alpha - (b - d))
    return(k)



def partial_derivative(func, var=0, point=[]):
    """ For calculating partial derivative of multi-variable function """

    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

# def stability_analysis_fnbd(params):
#     """
#     """

#     Pstar, Hstar = equilibrium_fnbd(params)

#     pgf_part_P = partial_derivative(fnbd_pgf, 0, [Pstar, Hstar, params['k'], params['xi']])
#     pgf_part_H = partial_derivative(fnbd_pgf, 1, [Pstar, Hstar, params['k'], params['xi']])
#     pgf = fnbd_pgf(Pstar, Hstar, params['k'], params['xi'])

#     m = params['H0'] + Hstar
#     A11 = -params['d'] + params['b']*(pgf + Hstar*pgf_part_H)
#     A12 = -params['alpha'] + params['b']*Hstar*pgf_part_P

#     A21 = ((params['lam']*Pstar*m**-1) - \
#           (params['lam']*Pstar*Hstar*m**-2)) + \
#           ((params['alpha']*params['k']*(params['k'] + 1)*(Pstar - 1)*Pstar) / (Hstar*params['k'] + 1)**2)
#     A22 = -params['d'] - params['mu'] + (params['lam']*Hstar / m) - \
#         ((params['alpha']*(params['k']*(Hstar + 2*Pstar - 1) + 2*Pstar)) / (Hstar*params['k'] + 1))

#     # Using Routh-Hurwitz criteria
#     crit1 = (A11 + A22) < 0
#     crit2 = (A11*A22 > A12*A21)

#     # Return the jacobian
#     jac = np.matrix([[A11, A12], [A21, A22]])

#     return([(Pstar, Hstar), (crit1 and crit2), jac])

def stability_analysis_feasible(params):
    """
    Calculate the stability of the host-parasite model with feasible
    aggregation using the Routh-Hurwitz criteria 

    Parameters
    ----------
    params : dict
        Contains model parameters

    Returns
    -------
    : list
        Tuple with Pstar, Hstar and a boolean indicating whether or not the 
        Routh-Hurwitz stability conditions are met. If true, the equilibrium is
        locally stable.

    """

    # Compute the equilibrium
    Pstar, Hstar = equilibrium_feasible(params)

    # Feasible surface calc and partial derivative of feasible surface
    k = feasible_k(Pstar, Hstar)
    k_part_P = partial_derivative(feasible_k, 0, [Pstar, Hstar])
    k_part_H = partial_derivative(feasible_k, 1, [Pstar, Hstar])

    # Partial derivatives of the dH / dt with respect to P and H

    # With respect to H
    t_H = k * Hstar
    tp_H = k_part_H*Hstar + k
    const1 = (params['xi'])*Pstar
    m_H = t_H / (const1 + t_H)
    mp_H = (tp_H*(const1 + t_H) - t_H*tp_H) / (const1 + t_H)**2
    g_H = k
    gp_H = k_part_H

    A11 = params['b']*(m_H**g_H + Hstar*m_H**g_H*(gp_H * np.log(m_H) + \
                                            (g_H / m_H)*mp_H)) - params['d']

    # With respect to P
    g_P = k
    gp_P = k_part_P
    t_P = g_P * Hstar
    tp_P = gp_P * Hstar
    q_P = (params['xi'])*Pstar + t_P
    qp_P = (params['xi']) + tp_P
    m_P = t_P / q_P
    mp_P = (tp_P * q_P - t_P * qp_P) / (q_P)**2

    A12 = Hstar*params['b']*m_P**g_P * (gp_P * np.log(m_P) + \
                                        (g_P / m_P) * mp_P) - params['alpha']

    # Partial derivatives of dP / dt with respect to H and P
    A21 = (params['lam']*Pstar / (params['H0'] + Hstar)) * (1 - Hstar / (params['H0'] + Hstar)) + \
          params['alpha']*Pstar**2*((k**2 + k + Hstar*k_part_H) / (Hstar**2 * k**2))
    A22 = -(params['d'] + params['mu'] + params['alpha']) + \
           (params['lam']*Hstar) / (params['H0'] + Hstar) + \
           params['alpha']*Pstar*((-2*k**2 - 2*k - Pstar*k_part_P) / (Hstar*k**2))

    # Using Routh-Hurwitz criteria
    crit1 = (A11 + A22) < 0
    crit2 = (A11*A22 > A12*A21)

    # Return the jacobian
    jac = np.matrix([[A11, A12], [A21, A22]])

    return([(Pstar, Hstar), (crit1 and crit2), jac])


def stability_analysis_fixed(params):
    """
    Local stability analysis of the fixed k macroparasite model

    Parameters
    ----------
    params : dict
        Contains model parameters

    Returns
    -------
    : list
        Tuple with Pstar, Hstar and a boolean indicating whether or not the 
        Routh-Hurwitz stability conditions are met. If true, the equilibrium is
        locally stable.

    """

    k = params['k']
    b = params['b']
    d = params['d']
    xi = params['xi']
    lam = params['lam']
    H0 = params['H0']
    mu = params['mu']
    alpha = params['alpha']

    Pstar, Hstar = equilibrium_nbd(params)

    # dH / dt with respect to H
    kern = (k*Hstar) / ((xi)*Pstar + k*Hstar)
    kern2 = (k*((xi)*Pstar + k*Hstar) - k**2 * Hstar) / (((xi)*Pstar + k*Hstar)**2)

    A11 = b*kern**k + b*Hstar*(k*kern**(k - 1) * kern2) - d
    A12 = b*Hstar*(k*kern**(k - 1) * (-1*k*Hstar*((xi)*Pstar + k*Hstar)**-2 * (xi))) - alpha

    A21 = lam*Pstar*(H0 / (H0 + Hstar)**2) + alpha*Pstar**2 * (Hstar**-2) * (1 + k**-1)
    A22 = -(d + mu) + (lam*Hstar) / (H0 + Hstar) - alpha - 2*Pstar*(alpha / Hstar)*(1 + k**-1)

    crit1 = (A11 + A22) < 0
    crit2 = (A11*A22 > A12*A21)

    # Return the jacobian
    jac = np.matrix([[A11, A12], [A21, A22]])

    return([(Pstar, Hstar), (crit1 and crit2), jac])


def feasible_k(P=10, H=10):
    """
    Pass in a P and H an extract k value predicted by feasible set

    Uses the moment correct k estimate

    Parameters
    ----------
    P : float
        Total number of parasites
    H : float
        Total number of hosts

    Returns
    -------
    : float
        Aggregation parameter of the negative binomial
    """

    try:
        P_ind = np.where(P > INTERP_VALS)[0][-1] + 1

        if P_ind > (N - 1):
            P_ind = N - 1
    except IndexError:
        P_ind = 0

    try:
        H_ind = np.where(H > INTERP_VALS)[0][-1] + 1

        if H_ind > (N - 1):
            H_ind = N - 1
    except IndexError:
        H_ind = 0

    k = INTERP_GRID[P_ind, H_ind]

    return(k)

# def feasible_k_mle(P=10, H=10):
#     """
#     Pass in a P and H an extract k value predicted by feasible set

#     Uses the MLE k estimate

#     Parameters
#     ----------
#     P : float
#         Total number of parasites
#     H : float
#         Total number of hosts

#     Returns
#     -------
#     : float
#         Aggregation parameter of the negative binomial
#     """

#     try:
#         P_ind = np.where(P > INTERP_VALS)[0][-1] + 1

#         if P_ind > (N - 1):
#             P_ind = N - 1
#     except IndexError:
#         P_ind = 0

#     try:
#         H_ind = np.where(H > INTERP_VALS)[0][-1] + 1

#         if H_ind > (N - 1):
#             H_ind = N - 1
#     except IndexError:
#         H_ind = 0

#     k = INTERP_GRID_MLE[P_ind, H_ind]

#     return(k)

# def feasible_k_approx(P, H):
#     return(1.4 / (1 + 10.11 / (P / H)))

get_feasible_k = np.vectorize(feasible_k)

def norm_k(P=10, H=10):
    return(1)

def update_macroparasite(H, P, params, k_fxn, delta_t=0.01):
    """
    Update rule for macroparasite model with feasible k.

    Parameters
    ----------
    H : float
        Number of hosts
    P : float
        Number of parasites
    params : dict
        Dictionary of parameters
        a, b, alpha, lam, H0, mu
    k_fxn : fxn
        Take in two values and returns value of aggregation
    delta_t : float
        Time step. Default is 0.01

    Returns
    -------
    : tuple
        Updated P and H values
    """

    k = k_fxn(P, H) # Update k based on current P and H values

    # Host update
    H_up = H + (params['a'] - (params['b'] + params['nu'] * H)) * H * delta_t - \
                                        params['alpha'] * P * delta_t

    # Parasite update
    P_up = P + P * delta_t * \
        (params['lam'] * H / (params['H0'] + H) -
        (params['b'] + params['nu']*H + params['mu'] + params['alpha']) -
        params['alpha'] * (P / H) * ((k + 1) / k))

    return H_up, P_up

def simulate_macroparasite(time, H0, P0, params, k_fxn, delta_t=0.01):
    """
    Function uses the update rule to simulate macroparasite model using Euler's
    method.

    Parameters
    ----------
    time : float
        How long of a time interval you want to simulate
    H0 : float
        Initial number of hosts
    P0 : float
        Initial number of parasites
    params : dict
        Parameters for the model
    k_fxn : Python function
        Function takes in two arguments and returns a values for k.
    delta_t: float
        Euler time step for the model

    Returns
    -------
    : dict
        H : Host trajectory
        P: Parasite trajectory
        t: time values
    """

    time_vals = np.linspace(0, time, num=time / delta_t)

    Hs = np.empty(len(time_vals))
    Ps = np.empty(len(time_vals))

    # Set initial conditions
    Hs[0] = H0
    Ps[0] = P0

    for i in np.arange(1, len(time_vals)):

        tH, tP = update_macroparasite(Hs[i - 1], Ps[i - 1], params, k_fxn,
                                                                    delta_t)
        Hs[i] = tH
        Ps[i] = tP

    return {'H' : Hs, 'P': Ps, 't': time_vals}


def feasible_mixture(para_host_vec, samples=200, center="mean"):
    """
    Gives a feasible set mixture from a given parasite host vector.

    Parameters
    -----------
    para_host_vec : list of tuples
        Length of the list is the number of heterogeneities. Each tuple is
        (P, H) where P is parasites and H is hosts. e.g. [(100, 10), (20, 30)]

    samples : int
        Number of feasible set samples to take

    center : str
        Either "mean", "median", or "mode".  They give very similar answers.  The mean
        will guarantee that the mean of the returned predicted vector will
        equal the expected mean for all sample sizes.  This is not necessarily
        True for the median or mode for small sample sizes. For the mode, if
        multiple values are found for the mode the minimum value is taken. The
        mode measure is more dependent on sample size than the mean or median,
        though it is what is used in Locey and White 2013.

    Returns
    -------
    : 2D array, 1D array
        The sorted array of all the sampled feasible sets
        The predicted center of the feasible set ("mean", "median", "mode")

    Examples
    --------
    full_feas, cent_feas = feasible_mixture([(100, 10), (20, 30)],
                                samples=200,  center="median")

    """

    mixed_sample = []
    para_host_vec = convert_to_int(para_host_vec)

    for ph in para_host_vec:

        if ph[0] == 0: # Hosts don't have any parasites

            tfeas = np.zeros(ph[1] * samples).reshape(samples, ph[1])
            mixed_sample.append(tfeas)

        else:

            tfeas = pyp.rand_partitions(int(ph[0]), int(ph[1]), samples, zeros=True)
            mixed_sample.append(tfeas)

    mix_feas = np.concatenate(mixed_sample, axis=1)
    sorted_feas = np.sort(mix_feas)[:, ::-1]

    if center == "mean":
        med_feas = np.mean(sorted_feas, axis=0)
    elif center == "median":
        med_feas = np.median(sorted_feas, axis=0)
    else:
        med_feas = stats.mode(sorted_feas, axis=0)[0].flatten()

    return sorted_feas, med_feas

def convert_to_int(para_host_vec):
    """
    Takes in a list of tuples and makes sure every item is a integer
    """

    para, host = zip(*para_host_vec)
    H = np.sum(host)
    para_round = np.round(para, decimals=0)
    host_round = np.round(host, decimals=0)

    # Make sure the vector still adds up. Should only miss by one (need to test)
    if np.sum(host) < H:
        ind = np.argmax(np.array(host) - np.floor(host))
        hosts_round[ind] = host_round[ind] + 1

    elif np.sum(host) > H:
        ind = np.argmin(np.array(host) - np.floor(host))
        hosts_round[ind] = host_round[ind] - 1


    return(zip(para_round.astype(np.int), host_round.astype(np.int)))


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
        
    coords = np.array([point.coords[0]
                       for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

