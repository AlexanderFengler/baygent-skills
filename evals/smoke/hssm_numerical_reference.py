"""Independent numerical oracles for the flat DDM validation, without sampling.

The Cython HDDM reference is a separate implementation from HSSM's analytical
PyTensor likelihood, but shares the Wiener model's mathematical ancestry.
Closed-form choice masses and a Brownian survival bound provide additional
checks. HSSM uses half the boundary separation used by HDDM (diffusion SD 1).
This module also supports offline checks of saved posterior likelihood entries.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def wiener_log_density(
    rt: ArrayLike,
    response: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    z: ArrayLike,
    t: ArrayLike,
) -> NDArray[np.float64]:
    """Return joint log p(RT, choice) through the direct, unfloored Cython API."""
    from hddm_wfpt import wfpt

    arrays = np.broadcast_arrays(rt, response, v, a, z, t)
    shape = arrays[0].shape
    rt, response, v, a, z, t = (
        np.ascontiguousarray(value, dtype=np.float64).reshape(-1) for value in arrays
    )
    zeros = np.zeros(rt.size, dtype=np.float64)
    return wfpt.wiener_logp_array(
        x=rt * response,
        v=v,
        sv=zeros,
        a=2 * a,
        z=z,
        sz=zeros,
        t=t,
        st=zeros,
        err=1e-10,
        simps_err=1e-10,
        p_outlier=0,
    ).reshape(shape)


def upper_choice_probability(v: float, a: float, z: float) -> float:
    """Closed-form upper-bound hitting probability for a unit-diffusion DDM."""
    if v == 0:
        return z
    separation = 2 * a
    return float(np.expm1(-2 * v * separation * z) / np.expm1(-2 * v * separation))


def survival_tail_bound(decision_time: float, v: float, a: float) -> float:
    """Bound probability that neither absorbing boundary is hit by this time.

    For zero drift, the absolute odd-eigenfunction survival series is bounded
    by (4/pi)*exp(-c)/(1-exp(-8c)), c=pi**2*T/(2*L**2), L=2*a.
    The drift change of measure is at most exp(abs(v)*L-v**2*T/2) for every
    surviving path. This deliberately loose bound does not use either density
    implementation and is useful at the long finite cutoffs used in the tests.
    """
    separation = 2 * a
    c = np.pi**2 * decision_time / (2 * separation**2)
    return float(
        (4 / np.pi)
        * np.exp(abs(v) * separation - v**2 * decision_time / 2 - c)
        / (-np.expm1(-8 * c))
    )
