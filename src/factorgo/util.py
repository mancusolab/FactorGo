import os
from typing import Tuple

import jax

from factorgo import infer


def update_x64(state: bool) -> None:
    jax.config.update("jax_enable_x64", state)
    return


def set_platform(platform=None) -> None:
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.

    :param str platform: either 'cpu', 'gpu', or 'tpu'.
    """
    if platform is None:
        platform = os.getenv("JAX_PLATFORM_NAME", "cpu")
    jax.config.update("jax_platform_name", platform)
    return


def set_hyper(hyper: Tuple) -> None:
    """
    set user-specified hyper parameters in priors
    """
    infer.HyperParams.halpha_a = float(hyper[0])
    infer.HyperParams.halpha_b = float(hyper[1])
    infer.HyperParams.htau_a = float(hyper[2])
    infer.HyperParams.htau_b = float(hyper[3])
    infer.HyperParams.hbeta = float(hyper[4])
    return
