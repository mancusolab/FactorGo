from dataclasses import dataclass
from typing import NamedTuple, Tuple, Union

import jax.numpy as jnp
import jax.numpy.linalg as jnpla
import jax.scipy.special as scp
from jax import jit, random


@dataclass
class Options:
    """simple class to store options for stopping rule (tolerance)"""

    elbo_tol: float = 1e-3
    max_iter: int = 10000


class HyperParams:
    """simple class to store options for hyper-parameters"""

    halpha_a: float = 1e-5
    halpha_b: float = 1e-5
    htau_a: float = 1e-5
    htau_b: float = 1e-5
    hbeta: float = 1e-5


class ZState(NamedTuple):
    """simple class to store results of posterior mean and variance for Z"""

    Z_m: jnp.ndarray
    Z_var: jnp.ndarray


class WState(NamedTuple):
    """simple class to store results of posterior mean and variance for W"""

    W_m: jnp.ndarray
    W_var: jnp.ndarray


class MuState(NamedTuple):
    """simple class to store results of posterior mean and variance (scalar) for Mu"""

    Mu_m: jnp.ndarray
    Mu_var: Union[float, jnp.ndarray]


class AlphaState(NamedTuple):
    """simple class to store options for hyper-parameters and moments for Alpha"""

    phalpha_a: Union[float, jnp.ndarray]
    phalpha_b: jnp.ndarray
    Ealpha: jnp.ndarray
    Elog_alpha: jnp.ndarray


class TauState(NamedTuple):
    """simple class to store options for hyper-parameters and moments for Tau"""

    phtau_a: Union[float, jnp.ndarray]
    phtau_b: Union[float, jnp.ndarray]
    Etau: Union[float, jnp.ndarray]
    Elog_tau: Union[float, jnp.ndarray]


class JointState(NamedTuple):
    """simple class to store options for joint update of variables"""

    Z_m: jnp.ndarray
    Z_var: jnp.ndarray
    W_m: jnp.ndarray
    W_var: jnp.ndarray
    Mu_m: jnp.ndarray
    phalpha_b: jnp.ndarray
    Ealpha: jnp.ndarray
    Elog_alpha: jnp.ndarray


class AuxState(NamedTuple):
    """simple class to store options for auxilary parameters"""

    b: jnp.ndarray
    R: jnp.ndarray
    R_inv: jnp.ndarray


def get_init(key_init, k, dat, log, init_opt: str = "random") -> Tuple:
    """initialize matrix for inference
    only need to initialize parameters required for updating Z, mu and W.
    quantities not used for updating those parameters are initialized with zeros
    as placeholder, with shape for jit to allocate memory
    """

    n, p = dat.shape
    w_shape = (p, k)

    W_var_init = jnp.identity(k)

    if init_opt == "svd":
        U, D, Vh = jnpla.svd(dat, full_matrices=False)
        W_m_init = Vh[0:k, :].T
        log.info("Initialize W using tSVD.")
    elif init_opt == "random":
        key_init, key_w = random.split(key_init)
        W_m_init = random.normal(key_w, shape=w_shape)
    else:
        log.info(f"{init_opt} is not supported")
        exit()

    Mu_m = jnp.zeros((p,))

    Ealpha_init = jnp.repeat(HyperParams.halpha_a / HyperParams.halpha_b, k)

    Etau = HyperParams.htau_a / HyperParams.htau_b

    # key quantities and placeholders
    wstate = WState(W_m_init, W_var_init)
    mustate = MuState(Mu_m, jnp.array([0.0]))
    alphastate = AlphaState(
        jnp.array([0.0]), jnp.zeros((k,)), Ealpha_init, jnp.zeros((k,))
    )
    taustate = TauState(jnp.array([0.0]), jnp.array([0.0]), Etau, jnp.array([0.0]))

    return wstate, mustate, alphastate, taustate


def batched_trace(A):
    """calculate trace for each batched matrice"""
    # bii->b represents doing batched trace operations
    return jnp.einsum("bii->b", A)


def calc_MeanQuadForm(wstate, WtW, zstate, mustate, B, sampleN, sampleN_sqrt):
    """calculate quadratic term E[(Zscore - fitted)^2]"""
    # import pdb; pdb.set_trace()
    W_m, _ = wstate
    Z_m, Z_var = zstate
    Mu_m, Mu_var = mustate
    p, _ = W_m.shape

    term1 = jnp.sum(B * B)
    term2 = jnp.sum(sampleN) * (p * Mu_var + Mu_m.T @ Mu_m)
    term3 = jnp.sum(
        sampleN
        * (batched_trace(WtW @ Z_var) + jnp.einsum("ni,ik,nk->n", Z_m, WtW, Z_m))
    )
    term4 = 2 * jnp.sum((Mu_m.T @ W_m) @ (sampleN[:, jnp.newaxis] * Z_m).T)
    term5 = 2 * jnp.trace(sampleN_sqrt[:, jnp.newaxis] * ((B @ W_m) @ Z_m.T))
    term6 = 2 * jnp.sum((B @ Mu_m) * sampleN_sqrt)

    mean_quad_form = term1 + term2 + term3 + term4 - term5 - term6

    return mean_quad_form


def logdet(M):
    """calculate log determinant for each batched matrice"""
    return jnpla.slogdet(M)[1]


# Update Posterior Moments
def pZ_main(B, wstate, EWtW, mustate, taustate, sampleN, sampleN_sqrt):
    """update posterior moments for factor score Z
    :pZ_m: (n,k) posterior moments
    :pZ_var: (n,k,k) posterior kxk covariance matrice for each study i
    """
    W_m, _ = wstate
    Mu_m, _ = mustate
    _, _, Etau, _ = taustate

    n, p = B.shape
    _, k = W_m.shape

    pZ_var = jnpla.inv(
        ((Etau * EWtW)[:, :, jnp.newaxis] * sampleN).swapaxes(-1, 0) + jnp.eye(k)
    )
    Bres = jnp.reshape((B / sampleN_sqrt[:, None] - Mu_m) * Etau, (n, p, 1))
    pZ_m = (pZ_var @ (W_m.T @ Bres)).squeeze(-1) * sampleN[:, None]

    return ZState(pZ_m, pZ_var)


def pMu_main(B, wstate, zstate, taustate, sampleN, sampleN_sqrt):
    """update posterior moments for intercept Mu
    :pMu_m: (p,)
    :pMu_var: a scalar (shared by all snps)
    """
    W_m, _ = wstate
    Z_m, _ = zstate
    _, _, Etau, _ = taustate

    sum_N = jnp.sum(sampleN)
    pMu_var = 1 / (HyperParams.hbeta + Etau * sum_N)
    ZWt = Z_m @ W_m.T
    res_sum = jnp.sum(sampleN[:, None] * (B / sampleN_sqrt[:, None] - ZWt), axis=0)
    pMu_m = Etau * pMu_var * res_sum

    return MuState(pMu_m, pMu_var)


def pW_main(B, zstate, mustate, taustate, alphastate, sampleN, sampleN_sqrt):
    """update posterior moments for factor loading W
    :pW_m: pxk
    :pW_V: kxk covariance matrice shared by all snps
    """
    Z_m, Z_var = zstate
    Mu_m, _ = mustate
    _, _, Etau, _ = taustate
    _, _, Ealpha, _ = alphastate

    n, _ = Z_m.shape
    Bres = B / sampleN_sqrt[:, None] - Mu_m
    tmp = Z_var.T @ sampleN + (Z_m.T * sampleN) @ Z_m
    pW_V = jnp.linalg.inv(Etau * tmp + jnp.diag(Ealpha))
    pW_m = jnp.einsum(
        "ik,np,nk->pi", Etau * pW_V, Bres, Z_m * sampleN[:, None], optimize="greedy"
    )

    return WState(pW_m, pW_V)


def palpha_main(WtW, p):
    """update posterior moments for ARD parameter alpha
    :phalpha_a: a scalar shared by all k latent factors
    :phalpha_b: (k,)
    """

    phalpha_a = HyperParams.halpha_a + p * 0.5
    phalpha_b = HyperParams.halpha_b + 0.5 * jnp.diagonal(WtW)

    Ealpha = phalpha_a / phalpha_b
    Elog_alpha = scp.digamma(phalpha_a) - jnp.log(phalpha_b)

    return AlphaState(phalpha_a, phalpha_b, Ealpha, Elog_alpha)


def get_aux(zstate, wstate, EWtW, taustate, sampleN):
    """find auxillary parameters for transformation method"""
    pZ_m, pZ_var = zstate
    pW_m, pW_var = wstate
    _, _, Etau, _ = taustate
    n, k = pZ_m.shape
    p, _ = pW_m.shape

    # 1) find b
    psi_n = Etau * p * pW_var
    # !! this can be simplified
    Psi = jnp.broadcast_to(psi_n[jnp.newaxis, ...], (n, k, k)) * sampleN.reshape(
        (n, 1, 1)
    ) + jnp.eye(k)
    Psi_Z = (Psi @ pZ_m.reshape((n, k, 1))).squeeze(-1)
    b = jnpla.inv(jnp.sum(Psi, axis=0)) @ jnp.sum(Psi_Z, axis=0)

    # 2) find R
    EZtZ = jnp.sum(pZ_var, axis=0) + pZ_m.T @ pZ_m
    # use jnpla.eigh() due to gpu end complains about jnpla.eig()
    # the result should be close subject to different ordering
    Lambda2, U = jnpla.eigh(EZtZ / n, symmetrize_input=False)
    U_weight = U * jnp.sqrt(Lambda2)

    quad_W = U_weight.T @ EWtW @ U_weight
    _, V = jnpla.eigh(quad_W, symmetrize_input=False)

    R = U_weight @ V
    R_inv = V.T / jnp.sqrt(Lambda2) @ U.T

    return AuxState(b, R, R_inv)


def pjoint_main(zstate, wstate, EWtW, mustate, alphastate, auxstate):
    """jointly transform latent space"""
    pZ_m, pZ_var = zstate
    pW_m, pW_var = wstate
    pMu_m, pMu_var = mustate
    phalpha_a, phalpha_b, Ealpha, Elogalpha = alphastate
    b, R, R_inv = auxstate
    p, _ = pW_m.shape

    # 1) remove bias
    pZ_m_center = pZ_m - b
    pMu_m_center = pMu_m + pW_m @ b

    # 2) rotate: each row of pW_m (pxk) and each row of pZ_m (nxk)
    pW_m_rot = (R.T @ pW_m.T).T
    pW_var_rot = R.T @ pW_var @ R

    # rotate each of row of pZ_m (nxk)
    pZ_m_rot = (R_inv @ pZ_m_center.T).T
    pZ_var_rot = (R_inv @ pZ_var) @ R_inv.T

    EWtW_q = R.T @ EWtW @ R
    phalpha_b_rot = HyperParams.halpha_b + 0.5 * jnp.diag(EWtW_q)
    Ealpha_rot = phalpha_a / phalpha_b_rot
    Elog_alpha_rot = scp.digamma(phalpha_a.real) - jnp.log(phalpha_b_rot.real)

    zstate_new = ZState(pZ_m_rot, pZ_var_rot)
    wstate_new = WState(pW_m_rot, pW_var_rot)
    mustate_new = MuState(pMu_m_center, pMu_var)
    alphastate_new = AlphaState(phalpha_a, phalpha_b_rot, Ealpha_rot, Elog_alpha_rot)

    return wstate_new, zstate_new, mustate_new, alphastate_new


def ptau_main(mean_quad, n, p):
    """update posterior moments for global scaling parameter Tau"""
    phtau_a = HyperParams.htau_a + n * p * 0.5
    phtau_b = 0.5 * mean_quad + HyperParams.htau_b

    Etau = phtau_a / phtau_b
    Elog_tau = scp.digamma(phtau_a) - jnp.log(phtau_b)

    return TauState(phtau_a, phtau_b, Etau, Elog_tau)


# write function to call all updating function
@jit
def runVB(
    B,
    wstate_old,
    EWtW_old,
    mustate_old,
    alphastate_old,
    taustate_old,
    sampleN,
    sampleN_sqrt,
):
    """One updating step in the recursive loop"""
    n, p = B.shape

    zstate = pZ_main(
        B, wstate_old, EWtW_old, mustate_old, taustate_old, sampleN, sampleN_sqrt
    )
    mustate = pMu_main(B, wstate_old, zstate, taustate_old, sampleN, sampleN_sqrt)
    wstate = pW_main(
        B, zstate, mustate, taustate_old, alphastate_old, sampleN, sampleN_sqrt
    )
    EWtW = p * wstate.W_var + wstate.W_m.T @ wstate.W_m

    alphastate = palpha_main(EWtW, p)

    # find aux params b and R:
    auxstate = get_aux(zstate, wstate, EWtW, taustate_old, sampleN)
    wstate, zstate, mustate, alphastate = pjoint_main(
        zstate, wstate, EWtW, mustate, alphastate, auxstate
    )

    EWtW = p * wstate.W_var + wstate.W_m.T @ wstate.W_m

    mean_quad = calc_MeanQuadForm(
        wstate, EWtW, zstate, mustate, B, sampleN, sampleN_sqrt
    )
    taustate = ptau_main(mean_quad, n, p)

    return wstate, EWtW, zstate, mustate, alphastate, taustate, mean_quad


# ELBO functions
def KL_QW(W_m, W_var, Ealpha, Elog_alpha):
    """KL divergence between estimated posterior W and prior"""
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
    """KL divergence between estimated posterior Z and prior"""
    n, k = Z_m.shape
    kl_qz = 0.5 * jnp.sum(
        batched_trace(Z_var) + jnp.sum(Z_m * Z_m, axis=1) - k - logdet(Z_var)
    )
    return kl_qz


def KL_QMu(Mu_m, Mu_var):
    """KL divergence between estimated posterior Mu and prior"""
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
    """KL divergence between two gamma distributions"""
    kl_gamma = (
        (pa - ha) * scp.digamma(pa)
        - scp.gammaln(pa)
        + scp.gammaln(ha)
        + ha * (jnp.log(pb) - jnp.log(hb))
        + pa * ((hb - pb) / pb)
    )
    return kl_gamma


def KL_Qalpha(pa, pb):
    """KL divergence between estimated posterior Alpha and prior"""
    kl_qa = jnp.sum(KL_gamma(pa, pb, HyperParams.halpha_a, HyperParams.halpha_b))
    return kl_qa


def KL_Qtau(pa, pb):
    """KL divergence between estimated posterior Tau and prior"""
    kl_qtau = KL_gamma(pa, pb, HyperParams.htau_a, HyperParams.htau_b)
    return kl_qtau


@jit
def elbo(B, wstate, zstate, mustate, alphastate, taustate, mean_quad):
    """Calculate ELBO"""
    W_m, W_var = wstate
    Z_m, Z_var = zstate
    Mu_m, Mu_var = mustate
    phtau_a, phtau_b, Etau, Elog_tau = taustate
    phalpha_a, phalpha_b, Ealpha, Elog_alpha = alphastate
    n, p = B.shape

    pD = 0.5 * (n * p * Elog_tau - Etau * mean_quad)

    kl_qw = KL_QW(W_m, W_var, Ealpha, Elog_alpha)
    kl_qz = KL_QZ(Z_m, Z_var)
    kl_qmu = KL_QMu(Mu_m, Mu_var)
    kl_qa = KL_Qalpha(phalpha_a, phalpha_b)
    kl_qt = KL_Qtau(phtau_a, phtau_b)
    elbo_sum = pD - (kl_qw + kl_qz + kl_qmu + kl_qa + kl_qt)

    return (
        elbo_sum.real,
        pD.real,
        kl_qw.real,
        kl_qz.real,
        kl_qmu.real,
        kl_qa.real,
        kl_qt.real,
    )


# calculate R2 for ordered factors: exausted memory
def R2(B, W_m, Z_m, Etau, sampleN_sqrt):
    """Calculate variance explained by each inferred factor
    Use residuals to calculate this.
    """

    n, p = B.shape
    _, k = Z_m.shape

    tss = jnp.sum(B * B) * Etau

    sse = jnp.zeros((k,))
    for i in range(n):
        WZ = W_m * Z_m[i] * sampleN_sqrt[i]  # pxk
        res = B[i][:, None] - WZ  # pxk
        tmp = jnp.sum(res * res, axis=0)  # (k,)
        sse += tmp
    r2 = 1.0 - sse * Etau / tss

    return r2


@jit
def _inner_fit(
    B,
    wstate_old,
    EWtW_old,
    mustate_old,
    alphastate_old,
    taustate_old,
    sampleN,
    sampleN_sqrt,
):
    """update parameters and calculate ELBO"""
    wstate, EWtW, zstate, mustate, alphastate, taustate, mean_quad = runVB(
        B,
        wstate_old,
        EWtW_old,
        mustate_old,
        alphastate_old,
        taustate_old,
        sampleN,
        sampleN_sqrt,
    )

    check_elbo, pD, kl_qw, kl_qz, kl_qmu, kl_qa, kl_qt = elbo(
        B, wstate, zstate, mustate, alphastate, taustate, mean_quad
    )

    return wstate, EWtW, zstate, mustate, alphastate, taustate, check_elbo


def fit(B, args, k, key_init, log, options, sampleN, sampleN_sqrt):
    """Wrapper function for running factorgo"""
    # set initializers
    _, p_snps = B.shape

    log.info(f"Initializing mean parameters with seed {args.seed}.")
    wstate, mustate, alphastate, taustate = get_init(
        key_init, k, B, log, args.init_factor
    )
    EWtW = p_snps * wstate.W_var + wstate.W_m.T @ wstate.W_m

    log.info("Completed initialization.")
    f_finfo = jnp.finfo(float)  # Machine limits for floating point types
    oelbo, delbo = f_finfo.min, f_finfo.max

    log.info("Starting Variational inference.")
    log.info("first iter may be slow due to JIT compilation).")
    RATE = args.rate  # print per RATE iterations
    for idx in range(options.max_iter):

        wstate, EWtW, zstate, mustate, alphastate, taustate, check_elbo = _inner_fit(
            B, wstate, EWtW, mustate, alphastate, taustate, sampleN, sampleN_sqrt
        )

        delbo = check_elbo - oelbo
        oelbo = check_elbo
        if idx % RATE == 0:
            log.info(
                f"itr={idx} | Elbo = {check_elbo} | deltaElbo = {delbo} | Tau = {taustate.Etau}"
            )

        if jnp.fabs(delbo) < args.elbo_tol:
            break

    r2 = R2(B, wstate.W_m, zstate.Z_m, taustate.Etau, sampleN_sqrt)
    f_order = jnp.argsort(-jnp.abs(r2))  # get index for sorting
    ordered_r2 = r2[f_order]
    ordered_Z_m = zstate.Z_m[:, f_order]
    ordered_W_m = wstate.W_m[:, f_order]

    f_info = jnp.column_stack(
        (jnp.arange(k) + 1, alphastate.Ealpha.real[f_order], ordered_r2)
    )
    log.info(f"Finished inference after {idx} iterations.")
    log.info(f"Final elbo = {check_elbo} and resid precision Tau = {taustate.Etau}")
    log.info(f"Final sorted Ealpha = {jnp.sort(alphastate.Ealpha)}")
    log.info(f"Final sorted R2 = {ordered_r2}")

    return (
        wstate.W_m,
        wstate.W_var,
        zstate.Z_m,
        zstate.Z_var,
        f_info,
        f_order,
        ordered_W_m,
        ordered_Z_m,
    )
