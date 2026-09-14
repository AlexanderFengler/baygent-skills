"""Independent joint-density reference for the notebook likelihood handoff.

The Cython HDDM reference is a separate implementation from HSSM's analytical
PyTensor likelihood, but shares the Wiener model's mathematical ancestry.
HSSM uses half the boundary separation used by HDDM (diffusion SD 1).
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
