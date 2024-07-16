"""Motion in the image domain."""

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from ...phantom import DynamicData, Phantom
from ...simulation import SimConfig
from ..base import AbstractHandler
from .utils import add_motion, motion_generator


class RandomMotionImageHandler(AbstractHandler):
    """Add Random Motion in Image.

    Parameters
    ----------
    ts_std_mm
        Translation standard deviation, in mm/s.
    rs_std_mm
        Rotation standard deviation, in radians/s.

    motion_file: str
        If provided, the motion file is loaded and resampled to match the number
        of frames in the simulation. The motion is then added to the data.
    motion_file_tr: float
        Original TR of the motion file, in seconds.

    Notes
    -----
    The motion is generated by drawing from a normal distribution with standard
    deviation for the 6 motion parameters (3 translations and 3 rotations, in
    this order). Then the cumulative motion is computed by summing the motion
    at each frame.

    The handlers is parametrized with speed in mm/s and rad/s, as these values
    provides an independent control of the motion amplitude regardless of the
    time resolution for the simulation.
    """

    __handler_name__ = "motion-image"

    ts_std_mms: tuple[float, float, float] | None = None
    rs_std_degs: tuple[float, float, float] | None = None

    motion_file: str | None = None
    motion_file_tr_ms: float | None = None

    def __post_init__(self):
        if (self.ts_std_mms is None or self.rs_std_degs is None) and (
            self.motion_file is None or self.motion_file_tr is None
        ):
            raise ValueError(
                "At least one of ts_std_mm, rs_std_mm or motion_file must be provided."
            )
        self._motion_data = None
        if self.motion_file is not None:
            # load the motion file
            self._motion_data = np.loadtxt(self.motion_file)

    def get_dynamic(self, phantom: Phantom, sim_conf: SimConfig) -> DynamicData:
        """Get dynamic informations."""
        n_frames = sim_conf.max_n_shots

        if self._motion_data is not None:
            # resample the motion data to match the simulation framerate.
            motion = np.interp(
                np.arange(n_frames) * sim_conf.sim_tr_ms,
                np.arange(len(self._motion_data)) * self.motion_file_tr,
                self._motion_data,
            )
        else:
            ts_std_pix = np.array(self.ts_std_mms) / np.array(sim_conf.res_mm)
            motion = motion_generator(
                n_frames,
                ts_std_pix,
                self.rs_std_degs,
                sim_conf.sim_tr_ms / 1000,
                sim_conf.rng,
            )

        return DynamicData(
            name=self.__handler_name__,
            in_kspace=self.__is_kspace_handler__,
            data=motion.T,
            func=apply_motion_to_phantom,
        )


def apply_motion_to_phantom(
    phantom: Phantom, motions: NDArray, time_idx: int
) -> Phantom:
    """Apply motion to the phantom."""
    new_phantom = deepcopy(phantom)
    for i, tissue_mask in enumerate(new_phantom.tissue_masks):  # TODO Parallel ?
        new_phantom.tissue_masks[i] = add_motion(tissue_mask, motions[:, time_idx])
    return new_phantom
