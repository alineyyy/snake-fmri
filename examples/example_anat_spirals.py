"""
Compare Fourier Model and T2* Model for Stack of Spirals trajectory
===========================================

This examples walks through the elementary components of SNAKE.

Here we proceed step by step and use the Python interface. A more integrated
alternative is to use the CLI ``snake-main``

"""

# %%

# Imports
import numpy as np
from snake.simulation import SimConfig, default_hardware, GreConfig
from snake.phantom import Phantom
from snake.smaps import get_smaps
from snake.sampling import StackOfSpiralSampler
from snake.mrd_utils import make_base_mrd

# %%

sim_conf = SimConfig(
    max_sim_time=6,
    seq=GreConfig(TR=100, TE=30, FA=3),
    hardware=default_hardware,
    fov_mm=(181, 217, 181),
    shape=(60, 72, 60),
)
sim_conf.hardware.n_coils = 8
sim_conf.hardware.field_strength = 7
phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T")


# %%
# Setting up Acquisition Pattern and Initializing Result file.
# ------------------------------------------------------------

# The next piece of simulation is the acquisition trajectory.
# Here nothing fancy, we are using a stack of spiral, that samples a 3D
# k-space, with an acceleration factor AF=4 on the z-axis.

sampler = StackOfSpiralSampler(
    accelz=4, acsz=0.1, orderz="top-down", nb_revolutions=12, obs_time_ms=30
)

smaps = None
if sim_conf.hardware.n_coils > 1:
    smaps = get_smaps(sim_conf.shape, n_coils=sim_conf.hardware.n_coils)


# %%
# Acquisition with Cartesian Engine
# ---------------------------------
#
# The generated file ``example_EPI.mrd`` does not contains any k-space data for
# now, only the sampling trajectory. let's put some in. In order to do so, we
# need to setup the **acquisition engine** that models the MR physics, and get
# sampled at the specified k-space trajectory.
#
# SNAKE comes with two models for the MR Physics:
#
# - model="simple" :: Each k-space shot acquires a constant signal, which is the
#   image contrast at TE.
# - model="T2s" :: Each k-space shot is degraded by the T2* decay induced by
#   each tissue.

# Here we will use the "simple" model, which is faster.
#
# SNAKE's Engine are capable of simulating the data in parallel, by distributing
# the shots to be acquired to a set of processes. To do so , we need to specify
# the number of jobs that will run in parallel, as well as the size of a job.
# Setting the job size and the number of jobs can have a great impact on total
# runtime and memory consumption.
#
# Here, we have a single frame to acquire with 60 frames (one EPI per slice), so
# a single worker will do.

from snake.engine import NufftAcquisitionEngine

engine = NufftAcquisitionEngine(model="simple", snr=1000)

make_base_mrd("example_spiral.mrd", sampler, phantom, sim_conf, smaps=smaps)
make_base_mrd("example_spiral_t2s.mrd", sampler, phantom, sim_conf, smaps=smaps)

engine(
    "example_spiral.mrd",
    worker_chunk_size=60,
    n_workers=1,
    nufft_backend="stacked-gpunufft",
)
engine_t2s = NufftAcquisitionEngine(model="T2s", snr=1000)

engine_t2s(
    "example_spiral_t2s.mrd",
    worker_chunk_size=60,
    n_workers=1,
    nufft_backend="stacked-gpunufft",
)

# %%
# Simple reconstruction
# ---------------------
#
# Getting k-space data is nice, but
# SNAKE also provides rudimentary reconstruction tools to get images (and check
# that we didn't mess up the acquisition process).
# This is available in the companion package ``snake_toolkit``.
#
# Loading the ``.mrd`` file to retrieve all information can be done using the
# ``ismrmd`` python package, but SNAKE provides convient dataloaders, which are
# more efficient, and take cares of managing underlying files access. As we are
# showcasing the API, we will do things manually here, and use only core SNAKE.

from snake.mrd_utils import NonCartesianFrameDataLoader
from snake_toolkit.reconstructors import (
    SequentialReconstructor,
    ZeroFilledReconstructor,
)

zer_rec = ZeroFilledReconstructor(
    nufft_backend="stacked-gpunufft", density_compensation="pipe"
)
seq_rec = SequentialReconstructor(
    nufft_backend="stacked-gpunufft",
    density_compensation="pipe",
    max_iter_per_frame=50,
)
with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:
    adjoint_spiral = abs(zer_rec.reconstruct(data_loader, sim_conf)[0])
    cs_spiral = abs(seq_rec.reconstruct(data_loader, sim_conf)[0])
with NonCartesianFrameDataLoader("example_spiral_t2s.mrd") as data_loader:
    adjoint_spiral_T2s = abs(zer_rec.reconstruct(data_loader, sim_conf)[0])
    cs_spiral_T2s = abs(seq_rec.reconstruct(data_loader, sim_conf)[0])


# %%
# Plotting the result
# -------------------

import matplotlib.pyplot as plt
from snake_toolkit.plotting import axis3dcut

fig, axs = plt.subplots(2, 3, figsize=(30, 10))

for ax, img, title in zip(
    axs[0],
    (adjoint_spiral, adjoint_spiral_T2s, abs(adjoint_spiral - adjoint_spiral_T2s)),
    ("simple", "T2s", "diff"),
):
    axis3dcut(fig, ax, img.T, None, None, cbar=True, cuts=(40, 40, 40), width_inches=4)
    ax.set_title(title)


for ax, img, title in zip(
    axs[1],
    (cs_spiral, cs_spiral_T2s, abs(cs_spiral - cs_spiral_T2s)),
    ("simple", "T2s", "diff"),
):
    axis3dcut(fig, ax, img.T, None, None, cbar=True, cuts=(40, 40, 40), width_inches=4)
    ax.set_title(title + " CS")


plt.show()
