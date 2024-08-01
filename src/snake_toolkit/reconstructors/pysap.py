"""Reconstructors using PySAP-fMRI toolbox."""

import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Local imports
from snake.mrd_utils import (
    CartesianFrameDataLoader,
    MRDLoader,
    NonCartesianFrameDataLoader,
)
from snake.parallel import ArrayProps, SharedMemoryManager, array_from_shm, array_to_shm
from snake.simulation import SimConfig
from tqdm.auto import tqdm

from .base import BaseReconstructor
from .fourier import ifft


def _reconstruct_cartesian_frame(
    filename: os.PathLike,
    idx: int,
    smaps_props: ArrayProps | None,
    final_props: ArrayProps,
) -> int:
    """Reconstruct a single frame."""
    with (
        array_from_shm(final_props) as final_images,
        CartesianFrameDataLoader(filename) as data_loader,
    ):
        mask, kspace = data_loader.get_kspace_frame(idx)
        sim_conf = data_loader.get_sim_conf()
        adj_data = ifft(kspace, axis=tuple(range(len(sim_conf.shape), 0, -1)))
        if smaps_props is not None and data_loader.n_coils > 1:
            with array_from_shm(smaps_props) as smaps:
                adj_data_smaps_comb = np.sum(
                    abs(adj_data * smaps[0].conj()), axis=0
                ).astype(np.float32, copy=False)
        else:
            adj_data_smaps_comb = np.sum(abs(adj_data) ** 2, axis=0).astype(
                np.float32, copy=False
            )
        final_images[0][idx] = adj_data_smaps_comb
    return idx


class ZeroFilledReconstructor(BaseReconstructor):
    """Zero Filled Reconstructor."""

    __reconstructor_name__ = "adjoint"
    n_jobs: int = 10
    nufft_backend: str = "gpunufft"
    density_compensation: str | bool = "pipe"

    def setup(self, sim_conf: SimConfig) -> None:
        """Initialize Reconstructor."""
        pass

    def reconstruct(self, data_loader: MRDLoader, sim_conf: SimConfig) -> NDArray:
        """Reconstruct data with zero-filled method."""
        with data_loader:
            if isinstance(data_loader, CartesianFrameDataLoader):
                return self._reconstruct_cartesian(data_loader, sim_conf)
            elif isinstance(data_loader, NonCartesianFrameDataLoader):
                return self._reconstruct_nufft(data_loader, sim_conf)
            else:
                raise ValueError("Unknown dataloader")

    def _reconstruct_cartesian(
        self, data_loader: CartesianFrameDataLoader, sim_conf: SimConfig
    ) -> NDArray:
        smaps = data_loader.get_smaps()
        if smaps is None and data_loader.n_coils > 1:
            raise NotImplementedError("Missing coil combine code.")

        final_images = np.ones(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )

        with (
            SharedMemoryManager() as smm,
            ProcessPoolExecutor(self.n_jobs) as executor,
            tqdm(total=data_loader.n_frames) as pbar,
        ):
            smaps_props = None
            if smaps is not None:
                smaps_props, smaps_shared, smaps_sm = array_to_shm(smaps, smm)
            final_props, final_shared, final_sm = array_to_shm(final_images, smm)

            futures = {
                executor.submit(
                    _reconstruct_cartesian_frame,
                    data_loader._filename,
                    idx,
                    smaps_props,
                    final_props,
                ): idx
                for idx in range(data_loader.n_frames)
            }

            for future in as_completed(futures):
                future.result()
                pbar.update(1)
            final_images[:] = final_shared.copy()
            final_sm.close()
            if smaps_props is not None:
                smaps_sm.close()
            smm.shutdown()
        return final_images

    def _reconstruct_nufft(
        self, data_loader: NonCartesianFrameDataLoader, sim_conf: SimConfig
    ) -> NDArray:
        """Reconstruct data with nufft method."""
        from mrinufft import get_operator

        smaps = data_loader.get_smaps()

        traj, kspace_data = data_loader.get_kspace_frame(0)

        kwargs = dict(
            shape=data_loader.shape,
            n_coils=data_loader.n_coils,
            smaps=smaps,
        )
        print(self.density_compensation, type(self.density_compensation))
        if self.density_compensation is False:
            kwargs["density"] = None
        else:
            kwargs["density"] = self.density_compensation
        if "stacked" in self.nufft_backend:
            kwargs["z_index"] = "auto"
        nufft_operator = get_operator(
            self.nufft_backend,
            samples=traj,
            **kwargs,
        )

        final_images = np.empty(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )

        for i in tqdm(range(data_loader.n_frames)):
            traj, data = data_loader.get_kspace_frame(i)

            nufft_operator.samples = traj
            final_images[i] = abs(nufft_operator.adj_op(data))
        return final_images


# EOF
