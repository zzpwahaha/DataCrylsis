import numpy as np
import PhysicsConstants as mc
import uncertainties.unumpy as unp

def center():
    return None

def f(time, t_0, sigma_x_0, T):
    """
    Assumes Rubidium 87 to calculate sigma_v from T.
    Assumes that the imaging beam is large compared to the MOT to avoid extra laser profile overlapped with atom profile
    
    :param time: the (known) time of the measured waists.
    :param t_0: the (effective) offset of the times of the measured waists. This can compensate for
        the wait that often exists between release and measurement, as well as non-perfect release of the
        MOT.
    :param sigma_x: The initial waist of the MOT.
    :param T: The temperature of the MOT.
    """
    return  f_raw(time, t_0, sigma_x_0, T)

def f_raw(time, t_0, sigma_x_0, T):
    if T < 0:
        return 1e9*np.ones(len(time))
    if t_0 < 0:
        return 1e9*np.ones(len(time))
    sigma_v = np.sqrt(mc.k_B * T / mc.Rb87_M)
    sigma_xt = np.sqrt(sigma_v**2*(time+t_0)**2 + sigma_x_0**2)
    return sigma_xt

def f_unp(time, t_0, sigma_x_0, T):
    sigma_v = unp.sqrt(mc.k_B * T / mc.Rb87_M)
    sigma_xt = unp.sqrt(sigma_v**2*(time+t_0)**2 + sigma_x_0**2)
    return sigma_xt

def guess():
    """
    Returns guess values for the parameters of this function class based on the input. Used for fitting using this
    class.
    """
    return 0, 2e-4, 100e-6

def args():
    return ['t_0', 'sigma_x', 'T']
