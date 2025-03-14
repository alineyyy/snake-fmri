"""Conjugate Gradient descent solver."""

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from snake.mrd_utils import (
    CartesianFrameDataLoader,
    NonCartesianFrameDataLoader,
)

from .fourier import init_nufft
from .pysap import ZeroFilledReconstructor
from .pysap import RestartStrategy  

# def CG_Operator(operator, traj, shape):
#     """Conjugate Gradient descent solver."""
#     import scipy as sp
#     return sp.sparse.linalg.LinearOperator(
#         (np.prod(shape), np.prod(shape)),
#         matvec=lambda x: operator.op(x.reshape(shape)),
#         rmatvec=lambda x: operator.adj_op(operator.adj_op(x.reshape(shape)))
#     )

class ConjugateGradientReconstructor(ZeroFilledReconstructor):
    """Conjugate Gradient descent solver.

    Parameters
    ----------
    max_iter : int
            Maximum number of iterations.
    tol : float
            Tolerance for the solver.
    """

    __reconstructor_name__ = "cg"

    max_iter: int
    tol: float
    density_compensation: str | bool | None = False
    nufft_backend: str = "cufinufft"
    restart_strategy: RestartStrategy = RestartStrategy.REFINE

    def _reconstruct_nufft(self, data_loader: NonCartesianFrameDataLoader) -> NDArray:
        """Reconstruct the data using the NUFFT operator."""
        from mrinufft.extras.gradient import cg
    
        from scipy.optimize import minimize
        nufft_operator = init_nufft(
            data_loader,
            density_compensation=self.density_compensation,
            nufft_backend=self.nufft_backend,
        )

        final_images = np.empty(
            (data_loader.n_frames, *data_loader.shape), dtype=np.complex64
        )
        x_init_0 = np.zeros(data_loader.shape, dtype=np.complex64)

        traj, data = data_loader.get_kspace_frame(0)
        x_init = x_init_0.copy()
        x_iter = x_init.copy()
        pbar_frames = tqdm(total=data_loader.n_frames, position=0)
        loss_list = []
        for i, traj, data in data_loader.iter_frames():
            if data_loader.slice_2d:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )[0, :, :2]
                data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
                for j in range(data.shape[1]):
                    x_cg = minimize(
                        fun = lambda x, b=data[:,j]: np.linalg.norm(
                            nufft_operator.op(x.reshape(data_loader.shape[:2])) - b, 
                        )**2, 
                        x0=x_init[:,:,j].ravel(),
                        method="CG",
                        jac=lambda x, b=data[:,j]: nufft_operator.data_consistency(
                            x.reshape(data_loader.shape[:2]), b
                            ).ravel(),  
                        tol=self.tol,
                        options={"gtol": self.tol, "maxiter": self.max_iter}
                    )
                    loss_list.append(x_cg.fun)
                    x_iter[:,:,j] = x_cg.x.reshape(data_loader.shape[:2])
                    # x_iter[:,:,j],loss = cg(
                    #     nufft_operator, 
                    #     data[:,j], 
                    #     x_init=x_init[:,:,j], 
                    #     num_iter=self.max_iter, 
                    #     tol=self.tol
                    # )
                    # loss_list.append(loss)
            else:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                ) # fix: update traj when 3D
                x_iter, loss = cg(
                    nufft_operator, 
                    data, 
                    x_init=x_init, 
                    num_iter=self.max_iter, 
                    tol=self.tol
                )
                loss_list.append(loss)
            x_iter = x_iter.copy() 
            x_init = (
                    x_iter.copy()
                    if self.restart_strategy != RestartStrategy.COLD
                    else x_init.copy()
            )
            final_images[i, ...] = x_iter
        #final_images[0,...] = x_iter
            pbar_frames.update(1)
        if self.restart_strategy != RestartStrategy.REFINE:
            return final_images, loss_list
        pbar_frames.reset()
        x_init = x_iter.copy() 
        for i, traj, data in data_loader.iter_frames():
            if data_loader.slice_2d:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )[0, :, :2]
                data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
                for j in range(data.shape[1]):
                    x_cg = minimize(
                        fun = lambda x, b=data[:,j]: np.linalg.norm(
                            nufft_operator.op(x.reshape(data_loader.shape[:2])) - b, 
                        )**2, 
                        x0=x_init[:,:,j].ravel(),
                        method="CG",
                        jac=lambda x, b=data[:,j]: nufft_operator.data_consistency(
                            x.reshape(data_loader.shape[:2]), b).ravel(),  
                        tol=self.tol,
                        options={"maxiter": self.max_iter}
                    )
                    loss_list.append(x_cg.fun)
                    x_iter[:,:,j] = x_cg.x.reshape(data_loader.shape[:2])
                    # x_iter[:,:,j],loss = cg(
                    #     nufft_operator, 
                    #     data[:,j], 
                    #     x_init=x_init[:,:,j], 
                    #     num_iter=self.max_iter, 
                    #     tol=self.tol
                    # )
                    # loss_list.append(loss)
            else:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )
                x_iter, loss = cg(
                    nufft_operator, 
                    data, 
                    x_init=x_init,
                    num_iter=self.max_iter, 
                    tol=self.tol
                )
                loss_list.append(loss)

            final_images[i, ...] = x_iter
            #final_images[0, ...] = x_iter
            pbar_frames.update(1)
        return final_images, loss_list

    def _reconstruct_cartesian(self, data_loader: CartesianFrameDataLoader) -> NDArray:
        """Reconstruct the data for Cartesian Settings."""
        from mrinufft.extras.fft import (
            CartesianFourierOperator,
        )  # TODO this does not exists yet
        from mrinufft.extras.gradient import cg

        mask, data = data_loader.get_kspace_frame(0)
        nufft_operator = CartesianFourierOperator(mask, data_loader.shape)

        final_images = np.empty(
            (data_loader.n_frames, *data_loader.shape), dtype=np.float32
        )

        for i in tqdm(range(data_loader.n_frames)):
            traj, data = data_loader.get_kspace_frame(i)
            if data_loader.slice_2d:
                nufft_operator.samples = traj.reshape(
                    data_loader.n_shots, -1, traj.shape[-1]
                )[0, :, :2]
                data = np.reshape(data, (data.shape[0], data_loader.n_shots, -1))
                for j in range(data.shape[1]):
                    final_images[i, :, :, j] = cg(nufft_operator, data[:, j])
            else:
                final_images[i] = cg(
                    nufft_operator, data, num_iter=self.max_iter, tol=self.tol
                )
        return final_images