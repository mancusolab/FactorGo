import argparse as ap
import logging
import sys

import numpy as np

import jax.numpy as jnp
import jax.random as rdm

from . import infer, io, util


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


def _main(args):
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
        default=1e-3,
        type=float,
        help="Tolerance for change in tau to halt inference",
    )
    argp.add_argument(
        "--hyper",
        default=None,
        nargs="+",
        type=float,
        help="Input hyperparameter in order for alpha, tau, and beta",
    )
    argp.add_argument(
        "--max-iter",
        default=10000,
        type=int,
        help="Maximum number of iterations to learn parameters",
    )
    argp.add_argument(
        "--init-factor",
        choices=["random", "svd", "zero"],
        default="random",
        help="How to initialize the latent factors and weights",
    )
    argp.add_argument(
        "--scale",
        action="store_true",
        default=False,
        help="scale each SNPs effect across traits (Default=True)",
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
        "-o", "--output", type=str, default="factorgo", help="Prefix path for output"
    )

    args = argp.parse_args(args)  # a list a strings

    log = get_logger(__name__, args.output)
    if args.verbose:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    # setup to use either CPU (default) or GPU
    util.set_platform(args.platform)

    # ensure 64bit precision (default use 32bit)
    jax.config.update("jax_enable_x64", True)

    # init key (for jax)
    key = rdm.PRNGKey(args.seed)
    key, key_init = rdm.split(key, 2)  # split into 2 chunk

    log.info("Loading GWAS effect size and standard error.")
    B, sampleN, sampleN_sqrt = io.read_data(args.Zscore_path, args.N_path, log, args.scale)
    log.info("Finished loading GWAS effect size, sample size and standard error.")

    n_studies, p_snps = B.shape
    log.info(f"Found N = {n_studies} studies, P = {p_snps} SNPs")

    # number of factors
    k = args.k
    log.info(f"User set K = {k} latent factors.")

    # set optionas for stopping rule
    options = infer.Options(args.elbo_tol, args.tau_tol, args.max_iter)

    # set 5 hyperparameters: otherwise use default 1e-3
    if args.hyper is not None:
        HyperParams.halpha_a = float(args.hyper[0])
        HyperParams.halpha_b = float(args.hyper[1])
        HyperParams.htau_a = float(args.hyper[2])
        HyperParams.htau_b = float(args.hyper[3])
        HyperParams.hbeta = float(args.hyper[4])

    log.info(
        f"""set parameters
          {HyperParams.halpha_a},{HyperParams.halpha_b},
          {HyperParams.htau_a},{HyperParams.htau_b}, {HyperParams.hbeta}
        """
    )

    W_m, W_var, Z_m, Z_var, f_info, f_order, ordered_W_m, ordered_Z_m = infer.fit(B, args, k, key_init, log, n_studies,
                                                                            options, p_snps, sampleN, sampleN_sqrt)

    log.info("Writing results.")
    np.savetxt(f"{args.output}.Zm.tsv.gz", ordered_Z_m.real, fmt="%s", delimiter="\t")
    np.savetxt(f"{args.output}.Wm.tsv.gz", ordered_W_m.real, fmt="%s", delimiter="\t")
    np.savetxt(f"{args.output}.factor.tsv.gz", f_info, fmt="%s", delimiter="\t")

    # calculate E(W^2) [unordered]: W_m pxk, W_var kxk
    EW2 = W_m ** 2 + jnp.diagonal(W_var)
    # calculate E(Z^2) [unordered]: Z_m nxk, Z_var nxkxk
    EZ2 = np.zeros((n_studies, k))
    Z_m2 = Z_m ** 2
    for i in range(n_studies):
        EZ2[i] = Z_m2[i] + jnp.diagonal(Z_var[i])

    io.write_results(args.output, EW2, EZ2, W_var, f_order)

    log.info("Finished. Goodbye.")

    return 0


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))