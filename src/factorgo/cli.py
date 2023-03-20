import argparse as ap
import logging
import sys

import jax.random as rdm

from factorgo import infer, io, util


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
    argp.add_argument("-p", "--platform", choices=["cpu", "gpu", "tpu"], default="cpu")
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
    util.update_x64(True)

    # init key (for jax), split into 2 chunk
    key = rdm.PRNGKey(args.seed)
    key, key_init = rdm.split(key, 2)

    log.info("Loading GWAS effect size and standard error.")
    B, sampleN, sampleN_sqrt = io.read_data(
        args.Zscore_path, args.N_path, log, args.scale
    )
    log.info("Finished loading GWAS summary statistics and sample size.")

    n_studies, p_snps = B.shape
    log.info(f"Found N = {n_studies} studies, P = {p_snps} SNPs")

    # number of factors
    k = args.k
    log.info(f"User set K = {k} latent factors.")

    # set options for stopping rule
    options = infer.Options(args.elbo_tol, args.max_iter)

    # set 5 hyper-parameters: otherwise use default 1e-5
    if args.hyper is not None:
        util.set_hyper(args.hyper)

    log.info(
        f"""set parameters
          halpha_a: {infer.HyperParams.halpha_a},
          halpha_b: {infer.HyperParams.halpha_b},
          htau_a: {infer.HyperParams.htau_a},
          htau_b: {infer.HyperParams.htau_b},
          hbeta: {infer.HyperParams.hbeta}
        """
    )

    W_m, W_var, Z_m, Z_var, f_info, f_order, ordered_W_m, ordered_Z_m = infer.fit(
        B, args, k, key_init, log, options, sampleN, sampleN_sqrt
    )

    log.info("Writing results.")
    io.write_results(
        args.output, f_info, ordered_Z_m, ordered_W_m, W_var, Z_var, f_order
    )
    log.info("Finished. Goodbye.")

    return 0


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
