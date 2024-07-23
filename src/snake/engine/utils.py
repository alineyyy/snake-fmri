"""Utilities for the MRD format."""

import scipy as sp
import numpy as np
from numpy.typing import NDArray

from ..phantom import Phantom, PropTissueEnum
from ..simulation import SimConfig


def get_contrast_gre(phantom: Phantom, FA: float, TE: float, TR: float) -> NDArray:
    """Compute the GRE contrast at TE."""
    return (
        phantom.props[:, PropTissueEnum.rho]
        * np.sin(np.deg2rad(FA))
        * np.exp(-TE / phantom.props[:, PropTissueEnum.T2s])
        * (1 - np.exp(-TR / phantom.props[:, PropTissueEnum.T1]))
        / (
            1
            - np.cos(np.deg2rad(FA)) * np.exp(-TR / phantom.props[:, PropTissueEnum.T1])
        )
    )


def get_ideal_phantom(phantom: Phantom, sim_conf: SimConfig) -> NDArray:
    """Apply the contrast to the phantom and return volume."""
    contrast = get_contrast_gre(
        phantom, sim_conf.seq.FA, sim_conf.seq.TE, sim_conf.seq.TR
    )
    phantom_state = np.sum(
        phantom.masks * contrast[(..., *([None] * len(phantom.anat_shape)))],
        axis=0,
    )
    return phantom_state


def fft(image: NDArray, axis: tuple[int, ...] | int = -1) -> NDArray:
    """Apply the FFT operator.

    Parameters
    ----------
    image : array
        Image in space.
    axis : int
        Axis to apply the FFT.

    Returns
    -------
    kspace_data : array
        kspace data.
    """
    return sp.fft.ifftshift(
        sp.fft.fftn(sp.fft.fftshift(image, axes=axis), norm="ortho", axes=axis),
        axes=axis,
    )


def get_noise(chunk_data: NDArray, cov: NDArray, rng: np.random.Generator) -> NDArray:
    """Generate noise for a given chunk of k-space data."""
    n_coils = cov.shape[0]

    chunk_size, n_coils, *xyz = chunk_data.shape

    noise_shape = (2, *xyz[::-1], chunk_size)
    noise = np.ascontiguousarray(
        rng.multivariate_normal(np.zeros(n_coils), cov, size=noise_shape).T,
        dtype=np.float32,
    )
    if len(cov) == 1:
        noise = noise[0]  # work with single coil data.
    noise = noise.view(np.complex64)
    noise = noise[..., 0]
    noise = np.moveaxis(noise, 1, 0)
    return noise
