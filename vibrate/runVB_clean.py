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

from jax import random, jit

import jax.profiler

# server = jax.profiler.start_server(9999)
# jax.profiler.start_trace("/testres")

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
    hbeta: float = 1e-3


# class InitParams(NamedTuple):
#     """simple class to store options for initialization of matrix"""
# 
#     W_m: jnp.ndarray
#     W_var: jnp.ndarray
#     Mu_m: jnp.ndarray
#     Ealpha: jnp.ndarray  # 1d
#     Etau: Union[float, jnp.ndarray]  # for scalars

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
    W_var: Union[float, jnp.ndarray]


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

class JointState(NamedTuple):
    """simple class to store options for Joint update of variables"""

    Z_m_new: jnp.ndarray
    Z_var_new: jnp.ndarray
    W_m_new: jnp.ndarray
    W_var_new: jnp.ndarray
    Mu_m_new: jnp.ndarray
    phalpha_a_new: Union[float, jnp.ndarray]
    phalpha_b_new: jnp.ndarray
    Ealpha_new: jnp.ndarray
    Elog_alpha_new: jnp.ndarray

## read data
def read_data(z_path, N_path, log, removeN=False):
    """
    input z score summary stats: headers are ["snp", "trait1", "trait2", ..., "traitn"]
    input sample size file: one column of sample size (with header) which has the same order as above
    """

    # Read dataset (read as data frame)
    df_z = pd.read_csv(z_path, delimiter="\t", header=0)

    # drop the first column (axis = 1)
    snp_col = df_z.columns[0]
    df_z.drop(labels=[snp_col], axis=1, inplace=True)

    # convert str into numeric and transpose the data nxp
    df_z = df_z.astype("float").T

    # read sample size file and convert str into numerics
    df_N = pd.read_csv(N_path, delimiter="\t", header=0)
    df_N = df_N.astype("float")

    # convert to numpy/jax device-array (n,p)
    df_z = jnp.array(df_z)

    # convert sampleN (a file with one column and header)to arrays
    N_col = df_N.columns[0]
    sampleN = df_N[N_col].values
    sampleN_sqrt = jnp.sqrt(sampleN)
    
    if removeN:
        n,_ = df_z.shape
        sampleN = jnp.ones((n,))
        sampleN_sqrt = jnp.ones((n,))
        log.info("Remove N from model, set all N == 1.")

    return df_z, sampleN, sampleN_sqrt


def get_init(key_init, n, p, k, dat, log, init_opt="random"):
    """
    initialize matrix for inference
    We update moments for Z and W first, so here only initiaze parameters required for updating those
    """
    w_shape = (p, k)
    z_shape = (n, k)
    
    W_var_init = jnp.identity(k)
    Z_var_init = jnp.broadcast_to(jnp.identity(k)[jnp.newaxis, ...], (n, k, k))
        
    if init_opt == "svd":
        U, D, Vh = jnpla.svd(dat, full_matrices=False)
        W_m_init = Vh[0:k, :].T
        Z_m_init = U[:, 0:k] * D[0:k]
        log.info("Initialize W and Z using tsvd.")
    else: 
        key_init, key_w = random.split(key_init)
        W_m_init = random.normal(key_w, shape=w_shape)

        key_init, key_z = random.split(key_init)
        Z_m_init = random.normal(key_z, shape=z_shape)    
        

    Mu_m = jnp.zeros((p,))

    Ealpha_init = jnp.repeat(HyperParams.halpha_a / HyperParams.halpha_b, k)
    
    Etau = HyperParams.htau_a / HyperParams.htau_b

    return InitParams(
        W_m=W_m_init,
        W_var=W_var_init,
        Z_m=Z_m_init,
        Z_var=Z_var_init,
        Mu_m=Mu_m,
        Ealpha=Ealpha_init,
        Etau=Etau,
    )


## self defined function to do computation and keep batch
## note: not using this batched_WtVinvW() anymore b/c SE is dropped
# def batched_WtVinvW(W_m, W_var, sampleN):
#     # out WtVinvW: nxkxk
#     # each kxk only differ by a factor of N
#     return jnp.einsum(
#         "n,bik->nik", sampleN, W_var + batched_outer(W_m, W_m), optimize="greedy"
#     )


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
    p, _ = W_m.shape
    term1 = jnp.sum(B * B)
    term2 = jnp.sum(sampleN) * (p*Mu_var + Mu_m.T @ Mu_m)
    term3 = jnp.sum(sampleN * (
        batched_trace(WtW @ Z_var) + jnp.einsum("ni,ik,nk->n", Z_m, WtW, Z_m)
    ))
    term4 = 2 * jnp.sum((Mu_m.T @ W_m) @ (sampleN[:, jnp.newaxis] * Z_m).T)
    term5 = 2 * jnp.trace(sampleN_sqrt[:, jnp.newaxis] * ((B @ W_m) @ Z_m.T))
    term6 = 2 * jnp.sum((B @ Mu_m) * sampleN_sqrt)

    mean_quad_form = term1 + term2 + term3 + term4 - term5 - term6

    return mean_quad_form

def logdet(M):
    """
    calculate logdet for each batched matrice
    """
    return jnpla.slogdet(M)[1]


## Update Posterior Moments
@jit
def pZ_main(B, W_m, EWtW, Mu_m, Etau, sampleN, sampleN_sqrt):
    """
    :pZ_m: (n,k) posterior moments
    :pZ_var: (n,k,k) posterior kxk covariance matrice for each study i
    """
    n, p = B.shape
    _, k = W_m.shape
    pZ_var = jnpla.inv(
        ((Etau * EWtW)[:, :, jnp.newaxis] * sampleN).swapaxes(-1, 0) + jnp.eye(k)
    )
    Bres = jnp.reshape((B / sampleN_sqrt[:, None] - Mu_m) * Etau, (n, p, 1))
    pZ_m = (pZ_var @ (W_m.T @ Bres)).squeeze(-1) * sampleN[:, None]

    return ZState(pZ_m, pZ_var)


@jit
def pMu_main(B, W_m, Z_m, Etau, sampleN, sampleN_sqrt):
    """
    :pMu_m: (p,)
    :pMu_var: a scalar (shared by all snps)
    """
    sum_N = jnp.sum(sampleN)
    pMu_var = 1 / (HyperParams.hbeta + Etau * sum_N)
    ZWt = Z_m @ W_m.T
    res_sum = jnp.sum(sampleN[:, None] * (B / sampleN_sqrt[:, None] - ZWt), axis=0)
    pMu_m = Etau * pMu_var * res_sum

    return MuState(pMu_m, pMu_var)


@jit
def pW_main(B, Z_m, Z_var, Mu_m, Etau, Ealpha, sampleN, sampleN_sqrt):
    """
    :pW_m: pxk
    :pW_V: kxk covariance matrice shared by all snps
    """
    n, _ = Z_m.shape
    Bres = B / sampleN_sqrt[:, None] - Mu_m
    tmp = Z_var.T @ sampleN + (Z_m.T * sampleN) @ Z_m
    pW_V = jnp.linalg.inv(Etau * tmp + jnp.diag(Ealpha))
    pW_m = jnp.einsum(
        "ik,np,nk->pi", Etau * pW_V, Bres, Z_m * sampleN[:, None], optimize="greedy"
    )

    return WState(pW_m, pW_V)


@jit
def palpha_main(WtW, p):
    """
    :phalpha_a: shared by all k latent factirs
    :phalpha_b: (k,)
    """

    phalpha_a = HyperParams.halpha_a + p * 0.5
    phalpha_b = HyperParams.halpha_b + 0.5 * jnp.diagonal(WtW)

    Ealpha = phalpha_a / phalpha_b
    Elog_alpha = scp.digamma(phalpha_a) - jnp.log(phalpha_b)

    return AlphaState(phalpha_a, phalpha_b, Ealpha, Elog_alpha)

# @jit
def pjoint_main(B, pZ_m, pZ_var, pW_m, pW_var, pMu_m, phalpha_a, phalpha_b, Etau, sampleN):
    # jointly transform latent space
    n, k = pZ_m.shape
    p, _ = pW_m.shape

    # Auxillary parameter
    # 1) remove bias
    psi_n = Etau * p * pW_var
    # !! this can be simplified
    Psi = jnp.broadcast_to(psi_n[jnp.newaxis, ...], (n, k, k)) * sampleN.reshape((n,1,1)) + jnp.eye(k)
    Psi_Z = (Psi @ pZ_m.reshape((n,k,1))).squeeze(-1)
    b = jnpla.inv(jnp.sum(Psi, axis=0)) @ jnp.sum(Psi_Z, axis=0)
    
    # set b=0
    # b = jnp.zeros((k,))

    pZ_m_center = pZ_m - b
    pMu_m_center = pMu_m + pW_m @ b

    # 2) find R: (kxk) and R^-1
    # import pdb; pdb.set_trace()
    ZtZ = jnp.sum(pZ_var, axis=0) + pZ_m_center.T @ pZ_m_center # (k,k)
    WtW = p*pW_var + pW_m.T @ pW_m  # (k,k)

    # jnpla return complex128 output for 64-bit input; eigenvector on clumn of output
    # numpy.linalg.eig return complex64 for 32-bit input
    ZtZ_n = ZtZ/n
    Lambda2, U = jnpla.eig(ZtZ_n)
    U_weight = U @ jnp.diag(jnp.sqrt(Lambda2))
    # order columns by eigenvalue
    # U_order = jnp.argsort(Lambda2)
    # U_weight = U_weight[:, U_order]
    
    quad_W = U_weight.T @ WtW @ U_weight
    D, V = jnpla.eig(quad_W)

    # V_order = jnp.argsort(D)
    # V = V[:, V_order]
    
    R = U_weight @ V
    R_inv = jnpla.inv(R)
    # R_inv = V.T @ jnp.diag((1/jnp.sqrt(Lambda2)) @ U.T

    # rotate each row of pW_m (pxk)
    pW_m_rot = (R.T @ pW_m.T).T
    pW_var_rot = R.T @ pW_var @ R

    # rotate each of row of pZ_m (nxk)
    pZ_m_rot = (R_inv @ pZ_m_center.T).T
    pZ_var_rot = (R_inv @ pZ_var) @ R_inv.T
    
    WtW_q = R.T @ WtW @ R
    phalpha_b_rot = HyperParams.halpha_b + 0.5 * jnp.diag(WtW_q)
    Ealpha_rot = phalpha_a / phalpha_b_rot
    Elog_alpha_rot = scp.digamma(phalpha_b_rot.real) - jnp.log(phalpha_b_rot.real)

    return JointState(
        pZ_m_rot,
        pZ_var_rot,
        pW_m_rot,
        pW_var_rot,
        pMu_m_center,
        phalpha_a,
        phalpha_b_rot,
        Ealpha_rot,
        Elog_alpha_rot,
    )

@jit
def ptau_main(mean_quad, n, p):
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
        jnp.sum(HyperParams.hbeta * Mu_var)
        + HyperParams.hbeta * (Mu_m.T @ Mu_m)
        - p
        - p * jnp.log(HyperParams.hbeta)
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
def R2(B, W_m, Z_m, Etau, sampleN_sqrt, n, p):
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

    return elbo_sum.real, pD.real, kl_qw.real, kl_qz.real, kl_qmu.real, kl_qa.real, kl_qt.real


def main(args):
    argp = ap.ArgumentParser(description="")  # create an instance
    argp.add_argument("Zscore_path")
    argp.add_argument("N_path")
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
        "--hyper",
        default=None,
        nargs="+",
        type=float,
        help="Input hyperparameter in order for alpha, tau, and beta; Default hyperparameters are 1e-3",
    )
    argp.add_argument(
        "--max-iter",
        default=10000,
        type=int,
        help="Maximum number of iterations to learn parameters",
    )
    argp.add_argument(
        "--start-trans",
        default=10,
        type=int,
        help="Which iteration to start transformation",
    )
    argp.add_argument(
        "--init-factor",
        choices=["random", "svd", "zero"],
        default="random",
        help="How to initialize the latent factors and weights",
    )
    argp.add_argument(
        "--removeN",
        action="store_true",
        help="remove scalar N from model, i.e. set all N==1",
    )
    argp.add_argument(
        "--rate",
        default=250,
        type=int,
        help="Rate of printing elbo info; default is printing per 250 iters",
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
    B, sampleN, sampleN_sqrt = read_data(args.Zscore_path, args.N_path, log, args.removeN)
    log.info("Finished loading GWAS effect size, sample size and standard error.")

    n_studies, p_snps = B.shape
    log.info(f"Found N = {n_studies} studies, P = {p_snps} SNPs")

    # number of factors
    k = args.k
    log.info(f"User set K = {k} latent factors.")

    # set optionas for stopping rule
    options = Options(args.elbo_tol, args.tau_tol, args.max_iter)

    # set 5 hyperparameters
    if args.hyper is not None:
        # hyperlist = args.hyper.split()
        HyperParams.halpha_a = float(args.hyper[0])
        HyperParams.halpha_b = float(args.hyper[1])
        HyperParams.htau_a = float(args.hyper[2])
        HyperParams.htau_b = float(args.hyper[3])
        HyperParams.hbeta = float(args.hyper[4])
        log.info(f"set parameters {HyperParams.halpha_a},{HyperParams.halpha_b},{HyperParams.htau_a},{HyperParams.htau_b}, {HyperParams.hbeta} ")
        
    # set initializers
    log.info("Initalizing mean parameters.")
    (W_m, W_var, Z_m, Z_var, Mu_m, Ealpha, Etau) = get_init(key_init, n_studies, p_snps, k, B, log, args.init_factor)
    EWtW = p_snps * W_var + W_m.T @ W_m
    phalpha_a = HyperParams.halpha_a
    phalpha_b = HyperParams.halpha_b
    log.info("Completed initalization.")

    f_finfo = jnp.finfo(float)  ## Machine limits for floating point types.
    oelbo, delbo = f_finfo.min, f_finfo.max
    # oelbo_rot, delbo_rot = f_finfo.min, f_finfo.max # for debug
    otau, dtau = 1000, 1000  ## initial value for delta tau

    log.info(
        "Starting Variational inference (first iter may be slow due to JIT compilation)."
    )
    RATE = args.rate #250  # print per 250 iterations
    for idx in range(options.max_iter):
        # if (idx < args.start_trans):

        # import pdb; pdb.set_trace()
        # log.info(f"itr =  {idx}| update Z")
        # Z_m, Z_var, W_m, W_var, Mu_m, phalpha_a, phalpha_b, Ealpha, Elog_alpha = pjoint_main(B, Z_m, Z_var, W_m, W_var, Mu_m, phalpha_a, phalpha_b, Etau, sampleN)
        Z_m, Z_var = pZ_main(B, W_m, EWtW, Mu_m, Etau, sampleN, sampleN_sqrt)
        # log.info(f"itr =  {idx}| update Mu")
        Mu_m, Mu_var = pMu_main(B, W_m, Z_m, Etau, sampleN, sampleN_sqrt)
        # log.info(f"itr =  {idx}| update W")
        W_m, W_var = pW_main(B, Z_m, Z_var, Mu_m, Etau, Ealpha, sampleN, sampleN_sqrt)
        
        EWtW = p_snps * W_var + W_m.T @ W_m
        # log.info(f"itr =  {idx}| update alpha")
        phalpha_a, phalpha_b, Ealpha, Elog_alpha = palpha_main(EWtW, p_snps)
        
        # if (idx >= args.start_trans):
        #     Z_m, Z_var, W_m, W_var, Mu_m, phalpha_a, phalpha_b, Ealpha, Elog_alpha = pjoint_main(B, Z_m, Z_var, W_m, W_var, Mu_m, phalpha_a, phalpha_b, Etau, sampleN)
        #     EWtW = p_snps * W_var + W_m.T @ W_m
        
        mean_quad = calc_MeanQuadForm(W_m, EWtW, Z_m, Z_var, Mu_m, Mu_var, B, sampleN, sampleN_sqrt)
        # log.info(f"itr =  {idx}| update tau")
        phtau_a, phtau_b, Etau, Elog_tau = ptau_main(mean_quad, n_studies, p_snps)
        
        check_elbo, pD, kl_qw, kl_qz, kl_qmu, kl_qa, kl_qt  = elbo(
        W_m,W_var,Z_m,Z_var,Mu_m,Mu_var,phtau_a,phtau_b,phalpha_a,phalpha_b,Ealpha,Elog_alpha,Etau,Elog_tau,B,mean_quad
        )
        
        delbo = check_elbo - oelbo
        oelbo = check_elbo
        if idx % RATE == 0:
            log.info(
                # f"itr =  {idx} | Elbo = {check_elbo} | deltaElbo = {delbo} | Tau = {Etau}"
                f"itr =  {idx} | Elbo = {check_elbo} | deltaElbo = {delbo} | Tau = {Etau}| pD = {pD}|kl_qw={kl_qw}|kl_qz={kl_qz}|kl_qmu={kl_qmu}|kl_qa={kl_qa}|kl_qt={kl_qt}"
            )
            # W_m.block_until_ready()
            # jax.profiler.save_device_memory_profile(f"testres/testmemory{idx}.prof")
        if (idx >= args.start_trans):
            Z_m, Z_var, W_m, W_var, Mu_m, phalpha_a, phalpha_b, Ealpha, Elog_alpha = pjoint_main(B, Z_m, Z_var, W_m, W_var, Mu_m, phalpha_a, phalpha_b, Etau, sampleN)
            EWtW = p_snps * W_var + W_m.T @ W_m
            
        # if (idx >= args.start_trans):
        #     # import pdb; pdb.set_trace()
        #     Z_m, Z_var, W_m, W_var, Mu_m, phalpha_a, phalpha_b, Ealpha, Elog_alpha = pjoint_main(B, Z_m, Z_var, W_m, W_var, Mu_m, phalpha_a, phalpha_b, Etau, sampleN)
        #     EWtW = p_snps * W_var + W_m.T @ W_m
        #     mean_quad = calc_MeanQuadForm(W_m, EWtW, Z_m, Z_var, Mu_m, Mu_var, B, sampleN, sampleN_sqrt)
        #     phtau_a, phtau_b, Etau, Elog_tau = ptau_main(mean_quad, n_studies, p_snps)
            # check_elbo_rot, pD_rot, kl_qw_rot, kl_qz_rot, kl_qmu_rot, kl_qa_rot, kl_qt_rot = elbo(
            # W_m,W_var,Z_m,Z_var,Mu_m,Mu_var,phtau_a,phtau_b,phalpha_a,phalpha_b,Ealpha,Elog_alpha,Etau,Elog_tau,B,mean_quad
            # )
        # 
        #     delbo_rot = check_elbo_rot - oelbo_rot
        #     oelbo_rot = check_elbo_rot
        #     log.info(
        #         f"After rot: itr =  {idx} | Elbo = {check_elbo_rot} | deltaElbo = {delbo_rot} | Tau = {Etau}| pD = {pD_rot}|kl_qw={kl_qw_rot}|kl_qz={kl_qz_rot}|kl_qmu={kl_qmu_rot}|kl_qa={kl_qa_rot}|kl_qt={kl_qt_rot}"
        #     )

        dtau = Etau - otau
        otau = Etau

        if delbo < args.elbo_tol:
            break

    f_order = jnp.argsort(Ealpha)
    ordered_Z_m = Z_m[:, f_order]
    ordered_W_m = W_m[:, f_order]

    r2 = R2(B, W_m, Z_m, Etau, sampleN_sqrt, n_studies, p_snps)
    ordered_r2 = r2[f_order]

    f_info = np.column_stack((jnp.arange(k) + 1, jnp.sort(Ealpha), ordered_r2))

    log.info(f"Finished inference after {idx} iterations.")
    log.info(f"Final elbo = {check_elbo} and resid precision = {Etau}")
    log.info(f"Final sorted Ealpha = {jnp.sort(Ealpha)}")
    log.info(f"Final sorted R2 = {ordered_r2}")

    log.info("Writing results.")
    np.savetxt(f"{args.output}.Zm.tsv.gz", ordered_Z_m, fmt="%s", delimiter="\t")
    # np.savetxt(f"{args.output}.Mu.tsv.gz", Mu_m, fmt="%s", delimiter="\t")
    np.savetxt(f"{args.output}.Wm.tsv.gz", ordered_W_m, fmt="%s", delimiter="\t")
    np.savetxt(f"{args.output}.factor.tsv.gz", f_info, fmt="%s", delimiter="\t")

    log.info("Finished. Goodbye.")

    # check_elbo.block_until_ready()
    # jax.profiler.save_device_memory_profile("testres/testmemory.prof")

    return 0


# user call this script will treat it like a program
if __name__ == "__main__":
    sys.exit(
        main(sys.argv[1:])
    )  # grab all arguments; first arg is alway the name of the script
