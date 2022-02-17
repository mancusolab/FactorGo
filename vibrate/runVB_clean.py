#! /usr/bin/env python
import argparse as ap
import logging
import os
import sys

from dataclasses import dataclass
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.linalg as jspla
import jax.scipy.sparse.linalg as la
import jax.scipy.special as scp
import pandas as pd

import numpy as np

# import jax.numpy.linalg as jnpla
# import scipy.special as scp

from jax import random, jit

import jax.profiler

# server = jax.profiler.start_server(9999)
# jax.profiler.start_trace("/testres/tensorboard")

# disable jax preallocation or set the % of memory for preallocation
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'


def get_logger(name, path=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = 0
        console = logging.StreamHandler()
        logger.addHandler(console)

        log_format = "[%(asctime)s - %(levelname)s] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        console.setFormatter(formatter)

        if path is not None:
            disk_log_stream = open("{}.log".format(path), "w")
            disk_handler = logging.StreamHandler(disk_log_stream)
            logger.addHandler(disk_handler)
            disk_handler.setFormatter(formatter)

    return logger


## set platform for jax
def set_platform(platform=None):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.

    :param str platform: either 'cpu', 'gpu', or 'tpu'.
    """
    if platform is None:
        platform = os.getenv("JAX_PLATFORM_NAME", "cpu")
    jax.config.update("jax_platform_name", platform)
    return


@dataclass
class Options:
    """simple class to store options for stopping rule"""

    # tolerance
    elbo_tol: float = 1e-3
    tau_tol: float = 1e-3
    max_iter: int = 1000


class HyperParams:
    """simple class to store options for hyper-parameters"""

    halpha_a: float = 1e-3
    halpha_b: float = 1e-3
    htau_a: float = 1e-3
    htau_b: float = 1e-3
    beta: float = 1e-3


class InitParams(NamedTuple):
    W_m: jnp.ndarray
    W_var: jnp.ndarray
    Z_m: jnp.ndarray
    Z_var: jnp.ndarray
    Mu_m: jnp.ndarray
    Ealpha: jnp.ndarray  # 1d
    Etau: Union[float, jnp.ndarray]  # for scalars


class ZState(NamedTuple):
    """simple class to store results of orthogonalization + projection"""

    Z_m: jnp.ndarray
    Z_var: jnp.ndarray


class WState(NamedTuple):
    """simple class to store results of orthogonalization + projection"""

    W_m: jnp.ndarray
    W_var: jnp.ndarray


class MuState(NamedTuple):
    """simple class to store results of orthogonalization + projection"""

    Mu_m: jnp.ndarray
    Mu_var: jnp.ndarray


class AlphaState(NamedTuple):
    """simple class to store options for hyper-parameters and Ealpha"""

    phalpha_a: jnp.ndarray
    phalpha_b: jnp.ndarray
    Ealpha: jnp.ndarray
    Elog_alpha: jnp.ndarray


class TauState(NamedTuple):
    """simple class to store options for hyper-parameters and Etau"""

    phtau_a: jnp.ndarray
    phtau_b: jnp.ndarray
    Etau: Union[float, jnp.ndarray]
    Elog_tau: Union[float, jnp.ndarray]


## read data
def read_data(beta_z_path, N_path, var_path=None):

    # Read GE dataset (read as data frame)
    df_beta_z = pd.read_csv(beta_z_path, delimiter="\t", header=0)

    # drop the first column (axis = 1) and transpose the data nxp
    snp_col = df_beta_z.columns[0]
    df_beta_z = df_beta_z.drop(labels=[snp_col], axis=1).T

    # drop the subject id, convert str into numeric
    df_beta_z = df_beta_z.astype("float")

    if var_path is not None:
        df_var = pd.read_csv(var_path, delimiter="\t", header=0)
        df_var = df_var.drop(labels=[snp_col], axis=1).T
    else:
        n, p = df_beta_z.shape
        df_var = jnp.ones((n, p))

    # read sample size file and rename col
    df_N = pd.read_csv(N_path, delimiter="\t", header=0)
    df_N = df_N.astype("float")

    return df_beta_z, df_var, df_N


def get_init(key_init, B, k):
    # import pdb; pdb.set_trace()
    n, p = B.shape

    w_shape = (p, k)
    z_shape = (n, k)

    key_init, key_w = random.split(key_init)
    # W_var_init = jnp.broadcast_to(jnp.identity(k)[None, ...], (p, k, k))
    W_var_init = jnp.identity(k)
    W_m_init = random.normal(key_w, shape=w_shape)

    key_init, key_z = random.split(key_init)
    Z_var_init = jnp.broadcast_to(jnp.identity(k)[None, ...], (n, k, k))
    Z_m_init = random.normal(key_z, shape=z_shape)

    Mu_m = jnp.zeros((p,))

    Ealpha_init = jnp.repeat(HyperParams.halpha_a / HyperParams.halpha_b, k)
    ## initialize tau = 1
    return InitParams(
        W_m=W_m_init,
        W_var=W_var_init,
        Z_m=Z_m_init,
        Z_var=Z_var_init,
        Mu_m=Mu_m,
        Ealpha=Ealpha_init,
        Etau=1.0,
    )


# self defined function to do computation and keep batch
def batched_WtVinvW(W_m, W_var, sampleN):
    # out WtVinvW: nxkxk
    # each kxk only differ by a factor of N
    return jnp.einsum(
        "n,bik->nik", sampleN, W_var + batched_outer(W_m, W_m), optimize="greedy"
    )


def batched_outer(A, B):
    # bij,bjk->bik is batched outer product (outer product of each row)
    return jnp.einsum("bi,bk->bik", A, B)


def batched_inner(A, B):
    # bij,bij->b represents doing batched inner products
    return jnp.einsum("bij,bij->b", A, B)


def batched_trace(A):
    # bii->b represents doing batched trace operations
    return jnp.einsum("bii->b", A)


def batched_broadcast(A, B):
    # each element in A (1d array) times each row of B
    return jnp.einsum("i,ij->ij", A, B)


def calc_MeanQuadForm(W_m, WtW, Z_m, Z_var, Mu_m, Mu_var, B, sampleN, sampleN_sqrt):
    # import pdb; pdb.set_trace()
    ZWt = Z_m @ W_m.T  # nxp

    term1 = jnp.sum(B * B, axis=1)  # BtVinvB (n,)
    term2 = sampleN * jnp.sum(Mu_var + jnp.square(Mu_m))  # (n,)
    term3 = sampleN * (
        batched_trace(WtW @ Z_var) + jnp.einsum("ni,ik,nk->n", Z_m, WtW, Z_m)
    )
    term4_5 = 2 * sampleN * jnp.sum((Mu_m - B / sampleN_sqrt[:, None]) * ZWt, axis=1)
    term6 = 2 * jnp.dot(B * sampleN_sqrt[:, None], Mu_m)

    mean_quad_form = jnp.sum(term1 + term2 + term3 + term4_5 - term6)

    return mean_quad_form


def logdet(M):
    return jnpla.slogdet(M)[1]


## Update moments
@jit
def pZ_main(W_m, EWtW, Mu_m, Etau, B, sampleN, sampleN_sqrt):
    # pZ_m: (n,k)
    # pZ_var: (n,k,k)
    n, p = B.shape
    _, k = W_m.shape

    # nxkxk
    pZ_var = jnpla.inv(
        ((Etau * EWtW)[:, :, np.newaxis] * sampleN).swapaxes(-1, 0) + np.eye(k)
    )  # kxk

    # minus mu in shape (p,1)
    Bres = jnp.reshape((B / sampleN_sqrt[:, None] - Mu_m) * Etau, (n, p, 1))
    pZ_m = (pZ_var @ W_m.T @ Bres).squeeze(-1) * sampleN[:, None]

    return ZState(pZ_m, pZ_var)


@jit
def pMu_main(W_m, Z_m, Etau, B, sampleN, sampleN_sqrt):
    n, p = B.shape
    _, k = W_m.shape
    sum_N = jnp.sum(sampleN)
    # (p,)
    pMu_var = 1 / (HyperParams.beta + Etau * sum_N)
    # nxp
    ZWt = Z_m @ W_m.T
    res_sum = jnp.sum(sampleN[:, None] * (B / sampleN_sqrt[:, None] - ZWt), axis=0)
    pMu_m = Etau * pMu_var * res_sum

    return MuState(pMu_m, pMu_var)


@jit
def pW_main(Z_m, Z_var, Mu_m, Etau, Ealpha, B, sampleN, sampleN_sqrt):
    # minus mu
    n, _ = Z_m.shape
    Bres = B / sampleN_sqrt[:, None] - Mu_m  # nxp
    tmp = jnp.einsum(
        "n,nik->ik",
        sampleN,
        (Z_var + batched_outer(Z_m, Z_m)),
    )
    pW_V = jnp.linalg.inv(Etau * tmp + jnp.diag(Ealpha))
    pW_m = jnp.einsum(
        "ik,np,nk->pi", Etau * pW_V, Bres, Z_m * sampleN[:, None], optimize="greedy"
    )

    return WState(pW_m, pW_V)


@jit
def palpha_main(WtW, p):
    # p, k = W_m.shape

    phalpha_a = HyperParams.halpha_a + p * 0.5
    phalpha_b = HyperParams.halpha_b + 0.5 * jnp.diagonal(WtW)

    Ealpha = phalpha_a / phalpha_b
    Elog_alpha = scp.digamma(phalpha_a) - jnp.log(phalpha_b)

    return AlphaState(phalpha_a, phalpha_b, Ealpha, Elog_alpha)


@jit
def ptau_main(B, mean_quad):
    n, p = B.shape

    phtau_a = HyperParams.htau_a + n * p * 0.5
    phtau_b = 0.5 * mean_quad + HyperParams.htau_b

    Etau = phtau_a / phtau_b
    Elog_tau = scp.digamma(phtau_a) - jnp.log(phtau_b)

    return TauState(phtau_a, phtau_b, Etau, Elog_tau)


## ELBO functions
def KL_QW(W_m, W_var, Ealpha, Elog_alpha):
    p, k = W_m.shape
    kl_qw = -0.5 * jnp.sum(
        logdet(W_var)
        + k
        + jnp.sum(Elog_alpha)
        - jnp.trace(Ealpha * W_var)
        - jnp.sum(W_m * Ealpha * W_m, axis=1)
    )
    return kl_qw


def KL_QZ(Z_m, Z_var):
    n, k = Z_m.shape
    kl_qz = 0.5 * jnp.sum(
        batched_trace(Z_var) + jnp.sum(Z_m * Z_m, axis=1) - k - logdet(Z_var)
    )
    return kl_qz


def KL_QMu(Mu_m, Mu_var):
    p = Mu_m.size
    kl_qmu = 0.5 * (
        jnp.sum(HyperParams.beta * Mu_var)
        + HyperParams.beta * (Mu_m.T @ Mu_m)
        - p
        - p * jnp.log(HyperParams.beta)
        - jnp.sum(jnp.log(Mu_var))
    )
    return kl_qmu


def KL_gamma(pa, pb, ha, hb):
    kl_gamma = (
        (pa - ha) * scp.digamma(pa)
        - scp.gammaln(pa)
        + scp.gammaln(ha)
        + ha * (jnp.log(pb) - jnp.log(hb))
        + pa * ((hb - pb) / pb)
    )
    return kl_gamma


def KL_Qalpha(pa, pb):
    kl_qa = jnp.sum(KL_gamma(pa, pb, HyperParams.halpha_a, HyperParams.halpha_b))
    return kl_qa


def KL_Qtau(pa, pb):
    kl_qtau = KL_gamma(pa, pb, HyperParams.htau_a, HyperParams.htau_b)
    return kl_qtau


# calculate R2 for ordered factors
@jit
def R2(W_m, Z_m, Etau, B, sampleN_sqrt):
    n, p = B.shape

    tss = jnp.sum(B * B) * Etau
    resid = B.T - batched_outer(W_m.T, (Z_m * sampleN_sqrt[:, None]).T)
    sse = batched_trace(jnp.swapaxes(resid * Etau, -2, -1) @ resid)

    r2 = 1.0 - sse / tss

    return r2


@jit
def elbo(
    W_m,
    W_var,
    Z_m,
    Z_var,
    Mu_m,
    Mu_var,
    phtau_a,
    phtau_b,
    phalpha_a,
    phalpha_b,
    Ealpha,
    Elog_alpha,
    Etau,
    Elog_tau,
    B,
    mean_quad,
):
    n, p = B.shape
    # import pdb; pdb.set_trace()

    pD = 0.5 * (n * p * Elog_tau - Etau * mean_quad)
    kl_qw = KL_QW(W_m, W_var, Ealpha, Elog_alpha)
    kl_qz = KL_QZ(Z_m, Z_var)
    kl_qmu = KL_QMu(Mu_m, Mu_var)
    kl_qa = KL_Qalpha(phalpha_a, phalpha_b)
    kl_qt = KL_Qtau(phtau_a, phtau_b)
    elbo_sum = pD - (kl_qw + kl_qz + kl_qmu + kl_qa + kl_qt)

    return elbo_sum


def main(args):
    argp = ap.ArgumentParser(description="")  # create an instance
    argp.add_argument("beta_z_path")
    argp.add_argument("N_path")
    argp.add_argument(
        "--var-path",
        default=None,
        help="Add SE^2 of variants to calculate Z score",
    )
    argp.add_argument(
        "-k", type=int, default=10
    )  # "-" must only has one letter like "-k", not like "-knum"
    argp.add_argument(
        "--elbo-tol",
        default=1e-3,
        type=float,
        help="Tolerance for change in ELBO to halt inference",
    )
    argp.add_argument(
        "--tau-tol",
        default=1e-6,
        type=float,
        help="Tolerance for change in residual variance to halt inference",
    )
    argp.add_argument(
        "--max-iter",
        default=10000,
        type=int,
        help="Maximum number of iterations to learn parameters",
    )
    argp.add_argument(
        "--init-factor",
        choices=["random", "pca", "zero"],
        default="random",
        help="How to initialize the latent factors and weights",
    )
    argp.add_argument("-p", "--platform", choices=["cpu", "gpu"], default="cpu")
    argp.add_argument(
        "-s", "--seed", type=int, default=123456789, help="Seed for randomization."
    )
    argp.add_argument("-d", "--debug", action="store_true", default=False)
    argp.add_argument("-v", "--verbose", action="store_true", default=False)
    argp.add_argument(
        "-o", "--output", type=str, default="VBres", help="Prefix path for output"
    )

    args = argp.parse_args(args)  # a list a strings

    log = get_logger(__name__, args.output)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    # setup to use either CPU (default) or GPU
    set_platform(args.platform)

    # ensure 64bit precision (default use 32bit)
    jax.config.update("jax_enable_x64", True)

    # init key (for jax)
    key = random.PRNGKey(args.seed)
    key, key_init = random.split(key, 2)  # split into 2 chunk

    log.info("Loading GWAS effect size and standard error.")
    B, V, sampleN = read_data(args.beta_z_path, args.N_path, args.var_path)
    log.info("Finished loading GWAS effect size, sample size and standard error.")

    n_studies, p_snps = B.shape
    log.info(f"Found N = {n_studies} studies, P = {p_snps} SNPs")

    # convert to numpy/device-array (n,p)
    B = jnp.array(B)
    Vinv = 1 / jnp.array(V)

    # convert to Z score summary stats
    if args.var_path:
        B = B * jnp.sqrt(Vinv)
        # Vinv = jnp.ones((n_studies, p_snps))
        log.info(f"Calculate Z score summary stats using Beta/SE.")

    # convert sampleN to arrays
    N_col = sampleN.columns[0]
    sampleN = sampleN[N_col].values
    sampleN_sqrt = jnp.sqrt(sampleN)

    # number of factors
    k = args.k
    log.info(f"User set K = {k} latent factors.")

    # set hyper-parameters
    options = Options(args.elbo_tol, args.tau_tol, args.max_iter)

    # set initializers
    log.info("Initalizing mean parameters.")
    (W_m, W_var, _, _, Mu_m, Ealpha, Etau) = get_init(key_init, B, k)  # seed = 1
    EWtW = p * W_var + W_m.T @ W_m
    log.info("Completed initalization.")

    # reshape for inference
    n, p = B.shape

    f_finfo = jnp.finfo(float)  ## Machine limits for floating point types.
    oelbo, delbo = f_finfo.min, f_finfo.max
    otau, dtau = 1000, 1000  ## initial value for delta tau

    log.info(
        "Starting Variational inference (first iter may be slow due to JIT compilation)."
    )
    RATE = 250  # print per 250 iterations
    for idx in range(options.max_iter):

        # import pdb; pdb.set_trace()
        # Z: nxk, nxkxk
        Z_m, Z_var = pZ_main(W_m, EWtW, Mu_m, Etau, B, sampleN, sampleN_sqrt)

        # Mu: (p,), (p,)
        Mu_m, Mu_var = pMu_main(W_m, Z_m, Etau, B, sampleN, sampleN_sqrt)

        # W: pxk, kxk
        W_m, W_var = pW_main(Z_m, Z_var, Mu_m, Etau, Ealpha, B, sampleN, sampleN_sqrt)
        EWtW = p * W_var + W_m.T @ W_m
        phalpha_a, phalpha_b, Ealpha, Elog_alpha = palpha_main(EWtW, p_snps)

        mean_quad = calc_MeanQuadForm(
            W_m, EWtW, Z_m, Z_var, Mu_m, Mu_var, B, sampleN, sampleN_sqrt
        )

        phtau_a, phtau_b, Etau, Elog_tau = ptau_main(B, mean_quad)

        check_elbo = elbo(
            W_m,
            W_var,
            Z_m,
            Z_var,
            Mu_m,
            Mu_var,
            phtau_a,
            phtau_b,
            phalpha_a,
            phalpha_b,
            Ealpha,
            Elog_alpha,
            Etau,
            Elog_tau,
            B,
            mean_quad,
        )

        delbo = check_elbo - oelbo
        oelbo = check_elbo
        if idx % RATE == 0:
            log.info(
                f"itr =  {idx} | Elbo = {check_elbo} | deltaElbo = {delbo} | Tau = {Etau}"
            )

        dtau = Etau - otau
        otau = Etau

        if delbo < args.elbo_tol:
            break

    f_order = jnp.argsort(Ealpha)
    ordered_Z_m = Z_m[:, f_order]
    ordered_W_m = W_m[:, f_order]

    r2 = R2(W_m, Z_m, Etau, B, sampleN_sqrt)
    ordered_r2 = r2[f_order]

    f_info = np.column_stack((jnp.arange(k) + 1, jnp.sort(Ealpha), ordered_r2))

    log.info(f"Finished inference after {idx} iterations.")
    log.info(f"Final elbo = {check_elbo} and resid precision = {Etau}")
    log.info(f"Final sorted Ealpha = {jnp.sort(Ealpha)}")
    log.info(f"Final sorted R2 = {ordered_r2}")

    log.info("Writing results.")
    np.savetxt(f"{args.output}.Zm.tsv.gz", ordered_Z_m, fmt="%s", delimiter="\t")
    np.savetxt(f"{args.output}.Mu.tsv.gz", Mu_m, fmt="%s", delimiter="\t")
    np.savetxt(f"{args.output}.Wm.tsv.gz", ordered_W_m, fmt="%s", delimiter="\t")
    np.savetxt(f"{args.output}.factor.tsv.gz", f_info, fmt="%s", delimiter="\t")

    log.info("Finished. Goodbye.")

    # W_m.block_until_ready()
    # jax.profiler.save_device_memory_profile("testmemory.prof")

    return 0


# jax.profiler.stop_trace()

# user call this script will treat it like a program
if __name__ == "__main__":
    sys.exit(
        main(sys.argv[1:])
    )  # grab all arguments; first arg is alway the name of the script
