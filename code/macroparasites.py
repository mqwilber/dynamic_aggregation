import pandas as pd
from scipy.misc import derivative
import scipy.integrate as integrate
import numpy as np
import scipy.optimize as opt


## Global constants for approximating feasible set
z = 0.49
c = -0.33
b_slope = 0.57

def macroparasite_mean(t, y, p):
  """
  ODE macroparasite equation in terms of mean parasite load from
  Kretschmer and Adler 1993 assuming a negative binomial distribution.

  Parameters
  ----------
  t : float
    time
  y : list
    [Hosts, mean parasite load]
  p : dict
    Model parameters
      b, d, alpha, lam, H0, mu, k

  Notes
  -----
  For use with scipy.integrate.ode
  """

  H = y[0]
  x = y[1]
  
  rhsH = H*(p['b'] - p['d'] - p['alpha']*x)
  rhsx = x*( ((p['lam']*H) / (p['H0'] + H)) - 
              (p['b'] + p['mu']) - p['alpha']*(1 + x / p['k']))
  
  return([rhsH, rhsx])


def macroparasite_mean_repro(t, y, p):
  """
  ODE macroparasite equation in terms of mean parasite load from
  Kretschmer and Adler 1993 assuming a negative binomial distribution. 

  The model also includes a linear effect of parasite load on host reproduction
  following Anderson and May 1978.

  Parameters
  ----------
  t : float
    time
  y : list
    [Hosts, mean parasite load]
  p : dict
    Model parameters
      b, d, alpha, lam, H0, mu, k, xi

  Notes
  -----
  For use with scipy.integrate.ode
  """

  H = y[0]
  x = y[1]
  
  rhsH = H*(p['b'] - p['d'] - p['alpha']*x - p['xi']*x)
  rhsx = x*( ((p['lam']*H) / (p['H0'] + H)) - 
              (p['b'] + p['mu']) - p['alpha']*(1 + x / p['k']) + p['xi']*x)
  
  return([rhsH, rhsx])


def macroparasite_ph(t, y, p):
  """
  ODE macroparasite equation in terms of total hosts and total parasites
  as in Anderson and May 1978.

  Parameters
  ----------
  t : float
    time
  y : list
    [Hosts, Parasites]
  p : dict
    Model parameters
      b, d, alpha, lam, H0, mu, k

  Notes
  -----
  For use with scipy.integrate.ode
  """

  H = y[0]
  P = y[1]
  x = P / H
  
  rhsH = H*(p['b'] - p['d'] - p['alpha']*x)
  rhsP = -(p['d'] + p['mu'])*P + ((p['lam']*H) / (p['H0'] + H))*P - \
              p['alpha']*H*(x + x**2 * ((p['k'] + 1) / p['k']) )
  
  return([rhsH, rhsP])


def macroparasite_feasible(t, y, p):
  """
  ODE macroparasite equation in terms of mean parasite load from
  Kretschmer and Adler 1993. Assuming that the variance to mean ratio follows
  the predictions from a feasible set model as described in Wilber et al. 2017

  Parameters
  ----------
  t : float
    time
  y : list
    [Hosts, mean parasite load]
  p : dict
    Model parameters
      b, d, alpha, lam, H0, mu

  Notes
  -----
  For use with scipy.integrate.ode
  """
  H = y[0]
  x = y[1]
  
  rhsH = H*(p['b'] - p['d'] - p['alpha']*x)
  rhsx = x*( ((p['lam']*H) / (p['H0'] + H)) - 
              (p['b'] + p['mu']) - p['alpha']*fs_varmean(H, x))
  
  return([rhsH, rhsx])


def macroparasite_feasible_repro(t, y, p):
  """
  ODE macroparasite equation in terms of mean parasite load from
  Kretschmer and Adler 1993. Assuming that the variance to mean ratio follows
  the predictions from a feasible set model as described in Wilber et al. 2017

  Parameters
  ----------
  t : float
    time
  y : list
    [Hosts, mean parasite load]
  p : dict
    Model parameters
      b, d, alpha, lam, H0, mu, xi

  Notes
  -----
  For use with scipy.integrate.ode
  """
  H = y[0]
  x = y[1]
  
  rhsH = H*(p['b'] - p['d'] - p['alpha']*x - p['xi']*x)
  rhsx = x*( ((p['lam']*H) / (p['H0'] + H)) - 
              (p['b'] + p['mu']) - p['alpha']*fs_varmean(H, x) + p['xi']*x)
  
  return([rhsH, rhsx])


def fs_varmean(H, x):
  """
  The feasible set variance to mean ratio given H host and x mean parasites
  per hosts.

  Parameters
  ----------
  H : float
    Number of hosts
  x : float
    Mean parasite load

  Returns
  -------
  : float
    Variance to mean ratio

  Notes
  -----
  This is an approximation to the feasible set variance to mean ratio based on
  simulations from the exact feasible set
  """
  
  gH = c + b_slope*np.log(H)
  logvm = gH + np.log(x)*z

  return(np.exp(logvm))


def partial_fs_varmean(H, x):
  """
  Calculate the partial derivative of the approximate feasible set 
  variance to mean ratio function (fs_varmean)

  Parameters
  ----------
  H : float
    Number of hosts
  x : float
    Mean parasite load
  
  Returns
  -------
  : tuple
      (partial with respect to H, partial with respect to x)
  """
  
  c1 = np.exp(c)
  
  pi_H = x**z * c1 * b_slope * H**(b_slope - 1)
  pi_x = c1 * H**b_slope * z * x**(z - 1)
  return((pi_H, pi_x))


def feasible_rhs_H(H, x, p):
  """
  Right-hand side of feasible set macroparasite ODE for hosts (H)
  """

  rhsH = H*(p['b'] - p['d'] - p['alpha']*x)
  
  return(rhsH)


def feasible_rhs_x(H, x, p):
  """
  Right-hand side of feasible set macroparasite ODE for mean parasites (x)
  """

  rhsx = x*( ((p['lam']*H) / (p['H0'] + H)) - 
              (p['b'] + p['mu']) - p['alpha']*fs_varmean(H, x))
  
  return(rhsx)


def feasible_repro_rhs_H(H, x, p):
  """
  Right-hand side of feasible set macroparasite ODE for hosts (H) with repro 
  reduction
  """

  rhsH = H*(p['b'] - p['d'] - p['alpha']*x - p['xi']*x)
  
  return(rhsH)


def feasible_repro_rhs_x(H, x, p):
  """
  Right-hand side of feasible set macroparasite ODE for mean parasites (x)
  """

  rhsx = x*( ((p['lam']*H) / (p['H0'] + H)) - 
              (p['b'] + p['mu']) - p['alpha']*fs_varmean(H, x) + p['xi']*x)
  
  return(rhsx)


def mean_repro_rhs_H(H, x, p):
  """
  Right-hand side of mean macroparasite ODE for hosts (H) with repro 
  reduction
  """

  rhsH = H*(p['b'] - p['d'] - p['alpha']*x - p['xi']*x)
  
  return(rhsH)
  

def mean_repro_rhs_x(H, x, p):
  """
  Right-hand side of mean macroparasite ODE for mean parasites (x)
  """

  rhsx = x*( ((p['lam']*H) / (p['H0'] + H)) - 
              (p['b'] + p['mu']) - p['alpha']*(1 + (1 / p['k'])*x) + p['xi']*x)
  
  return(rhsx)

def regulation_boundary(params):
  """
  Lambda at the boundary of host regulation for Fixed k macroparasite model

  Returns
  -------
  : float
    value of lambda (parasite reproduction) at the regulation boundary

  """
  
  x_star = (params['b'] - params['d']) / params['alpha']
  pi_x = (1 + (1 / params['k'])*x_star)
  reg_lam = params['b'] + params['mu'] + params['alpha']*pi_x
  return(reg_lam)


def H_equil_fs(params, start=1, stop=100):
  """
  Host equilibrium under the feasible set macroparasite model

  Parameters
  ----------
  params : dict
    Parameters for model
  start, stop : float
    Upper and lower bounds on root solver

  Returns
  -------
  : float
    Equilibrium number of hosts
  """
    
  def zero_fxn(H, params):
      
      xstar = mean_equil(params)
      p1 = (params['lam']*H) / (params['H0'] + H)
      vm = fs_varmean(H, xstar)
      return(p1 - (params['b'] + params['mu'] + params['alpha']*vm))
  
  root = opt.brentq(zero_fxn, start, stop, args=(params,))
  #root = opt.fsolve(zero_fxn, x0=[start], args=(params,))
  return(root)

def H_equil_fs_repro(params, start=1, stop=100):
  """
  Host equilibrium under the feasible set macroparasite model

  Parameters
  ----------
  params : dict
    Parameters for model
  start, stop : float
    Upper and lower bounds on root solver

  Returns
  -------
  : float
    Equilibrium number of hosts
  """
    
  def zero_fxn(H, params):
      
      xstar = mean_equil_repro(params)
      p1 = ((params['lam']*H) / (params['H0'] + H)) + params['xi']*xstar
      vm = fs_varmean(H, xstar)
      return(p1 - (params['b'] + params['mu'] + params['alpha']*vm))
  
  root = opt.brentq(zero_fxn, start, stop, args=(params,))
  #root = opt.fsolve(zero_fxn, x0=[start], args=(params,))
  return(root)



def H_equil(params):
  """
  Host equilibrium based when hosts follow a NBD distribution with k = 1.

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
  return(H)


def H_equil_repro(params, start=1, stop=100):
  """
  Host equilibrium under the feasible set macroparasite model

  Parameters
  ----------
  params : dict
    Parameters for model
  start, stop : float
    Upper and lower bounds on root solver

  Returns
  -------
  : float
    Equilibrium number of hosts
  """
    
  def zero_fxn(H, params):
      
      xstar = mean_equil_repro(params)
      p1 = ((params['lam']*H) / (params['H0'] + H)) + params['xi']*xstar
      vm = (1 + (1 / params['k']) * xstar)
      return(p1 - (params['b'] + params['mu'] + params['alpha']*vm))
  
  root = opt.brentq(zero_fxn, start, stop, args=(params,))
  #root = opt.fsolve(zero_fxn, x0=[start], args=(params,))
  return(root)

    
def mean_equil(params):
  """
  Equilibrium mean number of parasites per host
  """
  return((params['b'] - params['d']) / params['alpha'])

def mean_equil_repro(params):
  """
  """

  return((params['b'] - params['d']) / (params['alpha'] + params['xi']))


def feasible_jacobian(params, start, stop):
  """
  Analytical solution to the feasible set ODE jacobian

  Parameters
  ----------
  params : dict
    Parameters for the ODE model
  start, stop : float
    Boundaries for the root solver

  Returns
  -------
  : 2 x 2 array
    Jacobian where A11 = d H(t) / dH, A12 = d H(t) / dx, A21 = d x(t) / dH, 
    A22 = d x(t) / dx at evaluated at the equilibrium.
  """
  
  H_star = H_equil_fs(params, start, stop)
  x_star = mean_equil(params)
  pi_H, pi_x = partial_fs_varmean(H_star, x_star)
  pi = fs_varmean(H_star, x_star)
  
  A11 = params['b'] - params['d'] - params['alpha']*x_star
  A12 = -params['alpha']*H_star
  
  A21 = ((x_star*params['lam']*params['H0']) / (params['H0'] + H_star)**2) - \
        params['alpha']*x_star * pi_H
  A22 = ((params['lam']*H_star) / (params['H0'] + H_star)) - (params['b'] + params['mu']) - \
          params['alpha']*(pi + x_star*pi_x)
      
  jac = np.array([[A11, A12], [A21, A22]])
  return(jac)


def feasible_repro_jacobian(params, start, stop):
  """
  Analytical solution to the feasible set ODE jacobian with linear reduction
  in reproduction.

  Parameters
  ----------
  params : dict
    Parameters for the ODE model
  start, stop : float
    Boundaries for the root solver

  Returns
  -------
  : 2 x 2 array
    Jacobian where A11 = d H(t) / dH, A12 = d H(t) / dx, A21 = d x(t) / dH, 
    A22 = d x(t) / dx at evaluated at the equilibrium.
  """
  
  H_star = H_equil_fs_repro(params, start, stop)
  x_star = mean_equil_repro(params)
  pi_H, pi_x = partial_fs_varmean(H_star, x_star)
  pi = fs_varmean(H_star, x_star)
  
  A11 = params['b'] - params['d'] - params['alpha']*x_star - params['xi']*x_star
  A12 = -params['alpha']*H_star - params['xi']*H_star
  
  A21 = ((x_star*params['lam']*params['H0']) / (params['H0'] + H_star)**2) - \
        params['alpha']*x_star * pi_H
  A22 = ((params['lam']*H_star) / (params['H0'] + H_star)) - \
          (params['b'] + params['mu']) - \
          params['alpha']*(pi + x_star*pi_x) + 2*params['xi']*x_star
      
  jac = np.array([[A11, A12], [A21, A22]])
  return(jac)
    

def mean_repro_jacobian(params, start, stop):
  """
  Jacobian for the mean macroparasite model with impact on reproduction


  """

  H_star = H_equil_repro(params, start, stop)
  x_star = mean_equil_repro(params)
  pi_x = (1 / params['k'])
  pi = (1 + (1 / params['k']) * x_star)
  
  A11 = params['b'] - params['d'] - params['alpha']*x_star - params['xi']*x_star
  A12 = -params['alpha']*H_star - params['xi']*H_star

  A21 = ((x_star*params['lam']*params['H0']) / (params['H0'] + H_star)**2)
  A22 = ((params['lam']*H_star) / (params['H0'] + H_star)) - \
          (params['b'] + params['mu']) - \
          params['alpha']*(pi + x_star*pi_x) + 2*params['xi']*x_star

  jac = np.array([[A11, A12], [A21, A22]])
  return(jac)


def partial_derivative(func, var=0, point=[]):
  """ For calculating partial derivative of multi-variable function """

  args = point[:]
  def wraps(x):
      args[var] = x
      return func(*args)
  return derivative(wraps, point[var], dx = 1e-6)